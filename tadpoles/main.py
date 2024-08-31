from typing import List, Any, Literal, Tuple
from itertools import count
from io import StringIO
import polars as pl


ITER_MAX = 10
NORM_LITS = Literal['explode', 'unnest', 'unnest-explode', 'unnest-first', 'explode-first']

field = pl.col("__standin__")

def model_starter(name: str, data: Any, expand: NORM_LITS = None) -> None:
    """Prints a basic Tadpoles model class with fields based on the provided data schema."""
    str_replace = [".", " ", "-", "/"]
    ldf = pl.LazyFrame(data, infer_schema_length=None)
    ldf = normalize(ldf, how=expand) if expand else ldf
    print(f"class {name}(Model):")
    for col, dtype in ldf.schema.items():
        field = col.lower()
        for char in str_replace:
            field = field.replace(char, "_")
        print(f'   {field}: pl.{dtype} = pl.col("{col}")')


def unnest_rename(ldf: pl.LazyFrame, columns: List[str], separator: str = ".") -> pl.LazyFrame:
    """Unnests and renames columns in the LazyFrame."""
    return ldf.with_columns(
        [
            pl.col(column).struct.rename_fields(
                [f"{column}{separator}{field.name}" for field in ldf.schema[column].fields]
            )
            for column in columns
        ]
    ).unnest(columns)


def get_expandable(ldf: pl.LazyFrame, how: str, columns: List[str] = None) -> Tuple[List[str], List[str]]:
    """Identifies expandable columns (structs and lists) based on the provided method."""
    if columns:
        columns = [col for col in ldf.columns if any(name in col for name in columns)]
    else:
        columns = ldf.columns
    structs = [name for name in columns if isinstance(ldf.schema[name], pl.Struct)] if 'unnest' in how else []
    lists = [name for name in columns if isinstance(ldf.schema[name], pl.List)] if 'explode' in how else []
    return structs, lists


def normalize(ldf: pl.LazyFrame, separator: str = ".", columns: List[str] = [], how: NORM_LITS = 'explode-unnest') -> pl.LazyFrame:
    """Normalizes the LazyFrame by expanding and unnesting columns."""
    structs, lists = get_expandable(ldf, how, columns)
    while structs or lists:
        if lists:
            ldf = ldf.explode(lists)
        else:
            ldf = unnest_rename(ldf, structs, separator)
        if 'first' in how:
            break
        structs, lists = get_expandable(ldf, how, columns)
    return ldf

pl.LazyFrame.normalize = normalize


class Field:
    """
    Represents a field in a Tadpoles model and must be given at least one Polars expression. It can also be used to provide a `default` value to fill `null` values,
    or to to flag a primary key column using `primary_key=True` which will add the column to the `py Model.primary_key` list in the model. 
    If a model needs to accept and transform data from different sources with different naming, multiple expressions can be provided to the `tadpoles.Field` object. 
    When deriving columns, the first of these expressions with matching source columns is evaluated.
    
    ## Example Usage
    In this example the `event_version` column will be evaluated from the source column `"message.event_type"` and all `null` values will be replaced with `'Action'`
    The `email` column will be evaluated from the first valid expression of the two, and the value of `Events.primary_key` will be `['email']`

    ```py
    from tadpoles import Model, Field, field

    class Events(Model):
        event_id: str
        timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
        event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
        event_version: str = Field(pl.col("message.event_version"), default='Action')
        email: str = Field(pl.col("message.metadata.email"), pl.col("event_details.user.email"), primary_key=True)
        event_flag: bool = pl.when(pl.col("event_type")=="login").then(True).otherwise(False)
    ```
    """
    def __init__(self, *values, primary_key: bool = False, **kwargs):
        self.expressions = []
        self.values = values if values else (None,)
        self.primary_key = primary_key
        self.name = None
        self.dtype = kwargs.get('dtype')
        self.default = kwargs.get('default')

    def __eq__(self, other) -> bool:
        return self.name == other

    def __lt__(self, other) -> bool:
        return self.name < other

    def __gt__(self, other) -> bool:
        return self.name > other

    def __str__(self) -> str:
        return f"Field(name={self.name}, expressions={self.expressions}, dtype={self.dtype})"

    @property
    def literal(self) -> pl.Expr:
        return pl.lit(self.default).alias(self.name).cast(self.dtype)

    def set_exprs(self) -> None:
        if not self.name:
            raise ValueError("No name attribute for field")
        for value in self.values:
            if value is None:
                self.expressions.append(pl.col(self.name).cast(self.dtype))
            elif not isinstance(value, pl.Expr):
                self.default = value
                self.expressions.append(pl.col(self.name).cast(self.dtype).fill_null(self.default))
            else:
                
                expr_str = value.meta.serialize(format='json').replace("__standin__", self.name)
                expr = pl.Expr.deserialize(StringIO(expr_str), format='json').alias(self.name)
                if self.dtype not in [pl.Struct, pl.List, list, dict]:
                    expr = expr.cast(self.dtype)
                self.expressions.append(expr)

    def derivable(self, context: List[str]) -> pl.Expr:
        for expr in self.expressions:
            if all(source in context for source in expr.meta.root_names()):
                return expr


class TadpoleBase(pl.LazyFrame):
    __iter_max__ = ITER_MAX
    fields: List[Field]

    def __init__(self, *args, derive: bool = True, expand: NORM_LITS = None, expand_columns: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if expand:
            self._ldf = self.normalize(how=expand, columns=expand_columns)._ldf
        self.derived_fields = []
        if derive:
            self.derive()

    def __getattribute__(self, name: str) -> Any:
        result = super().__getattribute__(name)
        if isinstance(result, pl.LazyFrame):
            self._ldf = result._ldf
            return self
        return result

    @classmethod
    def model_schema(cls) -> dict:
        return {field.name: field.dtype for field in cls.fields}

    def underived_fields(self) -> List[Field]:
        return [field for field in self.fields if field not in self.derived_fields]

    def derive(self) -> None:
        if not self.columns:
            self._ldf = pl.LazyFrame(schema=self.model_schema())._ldf
            return
        for n in count():
            if n > self.__iter_max__:
                raise RuntimeError(
                    f"Failed to derive columns {self.underived_fields()}. Exceeded maximum {self.__iter_max__} derivation iterations."
                )
            exprs = self.derivable_expressions()
            if not exprs:
                break
            self._ldf = self.with_columns(exprs)._ldf
        literals = [field.literal for field in self.underived_fields()]
        self._ldf = self.with_columns(literals).select(sorted([field.name for field in self.fields]))._ldf

    def derivable_expressions(self) -> List[pl.Expr]:
        exprs = []
        for field in self.underived_fields():
            expr = field.derivable(self.columns)
            if expr is not None:
                exprs.append(expr)
                self.derived_fields.append(field)
        return exprs


class TadpoleMeta(type):

    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: dict, **kwargs):
        fields: List[Field] = attrs.pop("__fields__", [])
        annotations: dict = attrs.get("__annotations__", {})
        model_attrs = {key: val for key, val in attrs.items() if not key.startswith("_") and not callable(val)}
        model_types = {key: val for key, val in annotations.items() if not key.startswith("_") and key != "df"}

        for col in set(model_attrs) | set(model_types):
            value = model_attrs.pop(col, None)
            dtype = model_types.get(col)
            field = value if isinstance(value, Field) else Field(value)
            field.dtype = dtype
            field.name = col
            field.set_exprs()
            fields.append(field)

        for base in bases:
            if hasattr(base, "fields"):
                fields.extend(base.fields)

        attrs["primary_key"] = [field.name for field in fields if field.primary_key]
        attrs["fields"] = fields
        return super().__new__(mcs, name, bases, attrs, **kwargs)


class Model(TadpoleBase, metaclass=TadpoleMeta):
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
