## Tadpoles

Tadpoles is a Python package that extends the functionality of the polars library to make data transformations more concise and readable. It introduces the ability to define tables and column expressions in a manner reminiscent of Pydantic, providing a convenient and intuitive interface for data manipulation.

### Features

- **Readable Dataframe Schemas & Transormation**: Tadpoles model classes are easy to write and understand, making complex polars dataframe transformations simpler to code.
  
- **Column Derivation**: Tadpoles derives columns based on specified expressions, streamlining the process of ingesting complex data structures. Column derivation is done in groups only for expressions with available sources, allowing derived that reference other derived columns.

- **Nothing but Polars**: Built as an extension of the Polars library, Tadpoles has no other dependencies. Tadpoles takes advantage of Polars laziness when deriving columns, compiling the full expression before collecting.

- **Multiple Expressions for One Column**: Add flexibility to your model by defining multiple expressions for a field, and Tadpoles will derive the first one with an available source column. This allows you to define a single model for ingesting data from multiple sources that may have differing schemas.


## Defining a model
Define a class inhereting from the ```tadpoles.Model``` class. Columns are defined by the name, type, and value of the class attributes. The ```field``` object acts as a placeholder for ```pl.col('name')``` where name is the class attribute name.
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

Events(data)

```
Equates to:
```py
pl.DataFrame(data).lazy().with_columns(
    pl.col("event_id").cast(str),
    pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").cast(pl.Datetime),
    pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True).alias("event_type")
    pl.col("message.event_version").alias("event_version").cast(str),
    pl.col("message.metadata.email").alias("email").cast(str)
    ).collect()

```

## Deriving columns from other columns
The ```event_flag``` column is derived from ```event_type``` after that column is derived from the original source. Tadpoles determines the source for each column expression by calling ```pl.Expr.meta.root_names()```. For example:


```py
from tadpoles import Model, Field, field

class Events(Model):
    event_id: str
    timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
    event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
    event_version: str = pl.col("message.event_version")
    email: str = pl.col("message.metadata.email")
    event_flag: bool = pl.when(pl.col("event_type")=="login").then(True).otherwise(False)

Events(data)

```
Equates to:

```py
pl.DataFrame(data).lazy().with_columns(
    pl.col("event_id").cast(str),
    pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").cast(pl.Datetime),
    pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True).alias("event_type")
    pl.col("message.event_version").alias("event_version").cast(str),
    pl.col("message.metadata.email").alias("email").cast(str)
    ).with_columns(
        pl.when(pl.col("event_type")=="login").then(True).otherwise(False)
    ).collect()

```
## Providing multiple column expressions
If a model needs to accept and transform data from different sources with different naming, multiple expressions can be provided to the ```tadpoles.Field``` object. When deriving columns, the first of these expressions with matching source columns is evaluated. The ```email``` column will be evaluated from the first valid expression of the two.

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

## Unnest data automatically
Make Tadpoles unnest ```pl.Struct``` columns before derivation, making ingest of nested dictionaries simple. Nested dictionary keys are separtated by ```.```.

```py
from tadpoles import Model, Field, field

data = [
    {
        'id': 3299,
        'type': 'member',
        'attributes': {
            'name': 'user1',
            'role': 'admin'
        }
    },
    {
        'id': 4903,
        'type': 'member',
        'attributes': {
            'name': 'user2',
            'role': 'editor'
        }
    },
    {
        'id': 4532,
        'type': 'visitor',
        'attributes': {
            'name': 'user3',
            'role': 'reader'
        }
    }
]

class Users(Model):
    user_id: int = pl.col("id")
    type: str
    name: str = pl.col("attributes.name")
    role: str = pl.col("attributes.role")
    email: str = pl.format("{}@tadpoles.com", pl.col("name"))
    
df = Users(data, unnest=True)

shape: (3, 5)
┌──────────────────┬───────┬────────┬─────────┬─────────┐
│ email            ┆ name  ┆ role   ┆ type    ┆ user_id │
│ ---              ┆ ---   ┆ ---    ┆ ---     ┆ ---     │
│ str              ┆ str   ┆ str    ┆ str     ┆ i64     │
╞══════════════════╪═══════╪════════╪═════════╪═════════╡
│ user1@tadpoles.com ┆ user1 ┆ admin  ┆ member  ┆ 3299    │
│ user2@tadpoles.com ┆ user2 ┆ editor ┆ member  ┆ 4903    │
│ user3@tadpoles.com ┆ user3 ┆ reader ┆ visitor ┆ 4532    │
└──────────────────┴───────┴────────┴─────────┴─────────┘

```
# tadpoles
