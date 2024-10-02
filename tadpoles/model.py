from typing import List, Tuple
import polars as pl
import os
from .transform import transform, normalize, NORM_LITS, ITER_MAX
from .field import Field


def scan_file(path: str):
    ext = os.path.splitext(path)[1]
    if ext == ".csv":
        return pl.scan_csv(path)
    if ext in [".pq", ".parquet"]:
        return pl.scan_parquet(path)
    else:
        raise NotImplementedError

class ModelMeta(type):
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: dict, **kwargs):
        fields = {}
        type_hints = attrs.get("__annotations__", {})
        attrs.update({key: None for key in type_hints.keys() if key not in attrs})

        for base in bases:
            if issubclass(base, _Model) and hasattr(base, "fields"):
                fields.update({field.name: field for field in base.fields})

        for key, value in attrs.items():
            if any([key.startswith("_"), callable(value), key in _Model.__dict__]):
                continue
            dtype = type_hints.get(key, pl.Unknown)
            col = Field.from_attributes(key, dtype, value)
            attrs[key] = col
            fields[key] = col

        attrs["fields"] = list(fields.values())
        return super().__new__(mcs, name, bases, attrs, **kwargs)


class _Model(object):
    expand: NORM_LITS = None
    expand_columns: list = None
    fields: List[Field]

    def __init__(
        self,
        *args,
        from_file: str = None,
        expand: NORM_LITS = None,
        expand_columns: list = None,
        **kwargs,
    ):
        self.expand = expand or self.expand
        self.expand_columns = expand_columns or self.expand_columns
        if from_file:
            lf = scan_file(from_file)
        elif isinstance(args[0], pl.LazyFrame):
            lf = args[0]
        else:
            lf = pl.LazyFrame(*args, **kwargs)
        self.lf = normalize(lf, how=self.expand, columns=self.expand_columns)

    def __repr__(self):
        return f"{self.__class__.__name__}(fields={self.fields})"

    def __add__(self, other: "Model"):
        if issubclass(other.__class__, _Model):
            obj = self.__copy__()
            obj.append(other.lf)
            return obj
        return NotImplemented

    def __copy__(self):
        obj = self.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    @classmethod
    def transform(cls, ldf: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        return transform(ldf, cls.fields)
    
    @property
    def primary_key(self) -> list:
        return [col.name for col in self.fields if col.primary_key]

    def append(self, other) -> None:
        "Add more data to the current model lazyframe"
        if issubclass(other.__class__, _Model):
            other_lf = other.lf
        elif isinstance(other, pl.LazyFrame):
            other_lf = other
        else:
            other_lf = self.__class__(other).lf
        self.lf = pl.concat([self.lf, other_lf], how="diagonal_relaxed")

    def collect(self, *args, **kwargs) -> pl.DataFrame:
        "Transform and collect transformed data as Polars DataFrame"
        lf = transform(self.lf, self.fields)
        return lf.collect(*args, **kwargs)


class Model(_Model, metaclass=ModelMeta):
    """
    A base class for defining Tadpoles models.
    Define attributes with type hints and Polars expressions to transform input data.
    Columns are defined by the name, type, and value of the class attributes.

    ## Example Usage
    The `tadpoles.field` object acts as a placeholder for `pl.col("name")` where `"name"` is equal to the attribute name.
    The transformation is evaluated lazily, to execute it and return a dataframe use the `collect` method.
    For example:

    ```py
    from tadpoles import Model, Field, field
    import polars as pl

    class Events(Model):
        event_id: str
        timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
        event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
        event_version: str = pl.col("message.event_version")
        email: str = pl.col("message.metadata.email")

    Events(data).collect()

    ```
    Equates to:
    ```py
    pl.LazyFrame(data).with_columns(
        pl.col("event_id").cast(str),
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").cast(pl.Datetime),
        pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True).alias("event_type")
        pl.col("message.event_version").alias("event_version").cast(str),
        pl.col("message.metadata.email").alias("email").cast(str)
        ).collect()

    ```

    ## Deriving columns from other columns
    The `event_flag` column is derived from `event_type` after the latter is derived from the original source.
    Tadpoles determines the source for each column expression by calling `pl.Expr.meta.root_names()`. For example:


    ```py
    from tadpoles import Model, Field, field

    class Events(Model):
        event_id: str
        timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
        event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
        event_version: str = pl.col("message.event_version")
        email: str = pl.col("message.metadata.email")
        event_flag: bool = pl.when(pl.col("event_type")=="login").then(True).otherwise(False)

    Events(data).collect()

    ```
    Equates to:

    ```py
    pl.LazyFrame(data).with_columns(
        pl.col("event_id").cast(str),
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").cast(pl.Datetime),
        pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True).alias("event_type")
        pl.col("message.event_version").alias("event_version").cast(str),
        pl.col("message.metadata.email").alias("email").cast(str)
        ).with_columns(
            pl.when(pl.col("event_type")=="login").then(True).otherwise(False)
        ).collect()

    ```
    ## Unnest and explode data structures automatically
    Tadpoles can unnest all `pl.Struct` and explode all `pl.List` columns before derivation using the `tadpoles.normalize` function, simplifying the extraction of nested dictionaries and lists.
    Set the `expand` keyword argument when instantiating the class to explode/unnest structured data.
    Nested dictionary keys are separtated by `.`. By default this will explode/unnest all columns, to limit normalization to specific columns, list them in the `expand_columns` keword argument.

    ```py
    from tadpoles import Model, Field, field
    import polars as pl

    data = [
        {
            "id": 3299,
            "type": "member",
            "attributes": {
                "name": "user1",
                "role": "admin",
                "companies": [
                    {
                        "name": "Stuff Co",
                        "id": 1234
                        },
                    {
                        "name": "Another Co",
                        "id": 5678
                        },

                ]
            }
        },
        {
            "id": 4903,
            "type": "member",
            "attributes": {
                "name": "user2",
                "role": "editor",
                "companies": [
                    {
                        "name": "Stuff Co",
                        "id": 1234
                        },
                    {
                        "name": "Another Co",
                        "id": 5678
                        },

                ]
            }
        },
        {
            "id": 4532,
            "type": "visitor",
            "attributes": {
                "name": "user3",
                "role": "reader",
                "companies": [
                    {
                        "name": "Stuff Co",
                        "id": 1234
                        },
                    {
                        "name": "Another Co",
                        "id": 5678
                        },

                ]
            }
        }
    ]

    class Users(Model):
        user_id: int = pl.col("id")
        type: str
        name: str = pl.col("attributes.name")
        role: str = pl.col("attributes.role")
        email: str = pl.format("{}@tadpoles.com", pl.col("name"))
        company_name: str = pl.col("attributes.companies.name")
        company_id: int = pl.col("attributes.companies.id")

    df = Users(data, expand="unnest-explode").collect()

    shape: (6, 7)
    ┌────────────┬──────────────┬────────────────────┬───────┬────────┬─────────┬─────────┐
    │ company_id ┆ company_name ┆ email              ┆ name  ┆ role   ┆ type    ┆ user_id │
    │ ---        ┆ ---          ┆ ---                ┆ ---   ┆ ---    ┆ ---     ┆ ---     │
    │ i64        ┆ str          ┆ str                ┆ str   ┆ str    ┆ str     ┆ i64     │
    ╞════════════╪══════════════╪════════════════════╪═══════╪════════╪═════════╪═════════╡
    │ 1234       ┆ Stuff Co     ┆ user1@tadpoles.com ┆ user1 ┆ admin  ┆ member  ┆ 3299    │
    │ 5678       ┆ Another Co   ┆ user1@tadpoles.com ┆ user1 ┆ admin  ┆ member  ┆ 3299    │
    │ 1234       ┆ Stuff Co     ┆ user2@tadpoles.com ┆ user2 ┆ editor ┆ member  ┆ 4903    │
    │ 5678       ┆ Another Co   ┆ user2@tadpoles.com ┆ user2 ┆ editor ┆ member  ┆ 4903    │
    │ 1234       ┆ Stuff Co     ┆ user3@tadpoles.com ┆ user3 ┆ reader ┆ visitor ┆ 4532    │
    │ 5678       ┆ Another Co   ┆ user3@tadpoles.com ┆ user3 ┆ reader ┆ visitor ┆ 4532    │
    └────────────┴──────────────┴────────────────────┴───────┴────────┴─────────┴─────────┘
    ```
    As a plain Polars statement this would look like:
    ```py

    df = (
        pl.LazyFrame(data)
        .unnest("attributes")
        .explode("companies")
        .with_columns(
            pl.col("companies").struct.rename_fields(["company_name", "company_id"])
        )
        .unnest("companies")
        .with_columns(pl.format("{}@tadpoles.com", pl.col("name")).alias("email"))
    ).collect()

    ```
    Expand options are as follows:
    ```py
    "unnest" # Unnests dictionaries
    "explode" # Explodes lists
    "unnest-explode" # Unnests and explodes dictionaries and lists
    "unnest-first" # Unnests only the first level of dictionaries
    "explode-first" # Explodes only the first level of lists

    ```
    ## Providing multiple column expressions
    If a model needs to accept and transform data from different sources with different naming, multiple expressions can be provided to the `tadpoles.Field` object.
    When deriving columns, the first of these expressions with matching source columns is evaluated. The `email` column will be evaluated from the first valid expression of the two.

    ```py
    from tadpoles import Model, Field, field

    class Events(Model):
        event_id: str
        timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
        event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
        event_version: str = pl.col("message.event_version")
        email: str = Field(pl.col("message.metadata.email"), pl.col("event_details.user.email"))
        event_flag: bool = pl.when(pl.col("event_type")=="login").then(True).otherwise(False)
    ```
    ## Subclasses and inheritence
    Column expressions from subclasses are inherited as if they were typical attributes. In this example the `Users` is a subclass of `People` and inherits its expressions.

    ```py
    class People(Model):
        type: str
        name: str = pl.col("attributes.name")
        email: str = pl.format("{}@tadpoles.com", pl.col("name"))

    df = People(data, normalize='unnest-explode').collect()

    shape: (6, 3)
    ┌────────────────────┬───────┬─────────┐
    │ email              ┆ name  ┆ type    │
    │ ---                ┆ ---   ┆ ---     │
    │ str                ┆ str   ┆ str     │
    ╞════════════════════╪═══════╪═════════╡
    │ user1@tadpoles.com ┆ user1 ┆ member  │
    │ user1@tadpoles.com ┆ user1 ┆ member  │
    │ user2@tadpoles.com ┆ user2 ┆ member  │
    │ user2@tadpoles.com ┆ user2 ┆ member  │
    │ user3@tadpoles.com ┆ user3 ┆ visitor │
    │ user3@tadpoles.com ┆ user3 ┆ visitor │
    └────────────────────┴───────┴─────────┘

    class Users(People):
        user_id: int = pl.col("id")
        role: str = pl.col("attributes.role")
        company_name: str = pl.col("attributes.companies.name")
        company_id: int = pl.col("attributes.companies.id")

    df = Users(data, normalize='unnest-explode').collect()

    shape: (6, 7)
    ┌────────────┬──────────────┬────────────────────┬───────┬────────┬─────────┬─────────┐
    │ company_id ┆ company_name ┆ email              ┆ name  ┆ role   ┆ type    ┆ user_id │
    │ ---        ┆ ---          ┆ ---                ┆ ---   ┆ ---    ┆ ---     ┆ ---     │
    │ i64        ┆ str          ┆ str                ┆ str   ┆ str    ┆ str     ┆ i64     │
    ╞════════════╪══════════════╪════════════════════╪═══════╪════════╪═════════╪═════════╡
    │ 1234       ┆ Stuff Co     ┆ user1@tadpoles.com ┆ user1 ┆ admin  ┆ member  ┆ 3299    │
    │ 5678       ┆ Another Co   ┆ user1@tadpoles.com ┆ user1 ┆ admin  ┆ member  ┆ 3299    │
    │ 1234       ┆ Stuff Co     ┆ user2@tadpoles.com ┆ user2 ┆ editor ┆ member  ┆ 4903    │
    │ 5678       ┆ Another Co   ┆ user2@tadpoles.com ┆ user2 ┆ editor ┆ member  ┆ 4903    │
    │ 1234       ┆ Stuff Co     ┆ user3@tadpoles.com ┆ user3 ┆ reader ┆ visitor ┆ 4532    │
    │ 5678       ┆ Another Co   ┆ user3@tadpoles.com ┆ user3 ┆ reader ┆ visitor ┆ 4532    │
    └────────────┴──────────────┴────────────────────┴───────┴────────┴─────────┴─────────┘
    ```
    """

    pass
