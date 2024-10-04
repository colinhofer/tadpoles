import polars as pl
from typing import List, Any
from io import StringIO

PLACEHOLDER = "__field__"

field = pl.col(PLACEHOLDER)

def root_replace(expr: pl.Expr, to_replace: str, new_root: str) -> pl.Expr:
    if to_replace in expr.meta.root_names():
        expr_str = expr.meta.serialize(format="json").replace(to_replace, new_root)
        expr = pl.Expr.deserialize(StringIO(expr_str), format="json")
    return expr

class Field:
    """
    Represents a field in a Tadpoles model. It can also be used to provide a `default` value to fill `null` values,
    or to to flag a primary key column using `primary_key=True` which will add the column to the `py Model.primary_key` list in the model. 
    If a model needs to accept and transform data from different sources with different naming, multiple expressions can be provided to the `tadpoles.Field` object. 
    When deriving columns, the first of these expressions with matching source columns is evaluated.
    
    ## Example Usage
    In this example the `event_version` column will be evaluated from the source column `"message.event_type"` and all `null` values will be replaced with `'Action'`
    The `email` column will be evaluated from the first valid expression of the two, and the value of `Events.primary_key` will be `['email']`

    ```py
    from tadpoles import Model, Field, field

    class Events(Model):
        event_id: str = Field(primary_key=True)
        timestamp: pl.Datetime = field.str.to_datetime("%Y-%m-%d %H:%M:%S")
        event_type: str = pl.col("message.event_type").str.replace_all("com.amazon.rum.", "", literal=True)
        event_version: str = Field(pl.col("message.event_version"), default='Action')
        email: str = Field(pl.col("message.metadata.email"), pl.col("event_details.user.email"))
        event_flag: bool = pl.when(pl.col("event_type")=="login").then(True).otherwise(False)
    ```
    """
    def __init__(
        self,
        *args,
        name: str = None,
        primary_key: bool = False,
        dtype: type = pl.Unknown,
        default: Any = None,
    ) -> None:
        self.exprs: List[pl.Expr] = []
        self.primary_key = primary_key
        self.dtype = dtype
        self.default = default
        self.values = args if args else (pl.col(PLACEHOLDER),)
        self.name = name
        self.derived = False
        self.prepared = False
        

    def __str__(self) -> str:
        return f"Field(name={self.name}, exprs={self.exprs}, dtype={self.dtype})"

    def __repr__(self) -> str:
        return f"Field(name={self.name}, exprs={self.exprs}, dtype={self.dtype})"
        
    @classmethod
    def from_attributes(cls, name: str, dtype: type, value: Any) -> "Field":
        if isinstance(value, cls):
            value.name = name
            value.dtype = dtype
            return value.prepare()
        else:
            return Field(value, name=name, dtype=dtype).prepare()

    @property
    def literal(self):
        return pl.lit(self.default).cast(self.dtype)

    def value_expr(self, value: Any):
        if isinstance(value, pl.Expr):
            expr = root_replace(value, PLACEHOLDER, self.name)
        else:
            self.default = value
            expr = pl.col(self.name)
        expr = expr.cast(self.dtype)
        return expr if not self.default else expr.fill_null(self.default)

    def prepare(self):
        if not self.name:
            raise ValueError("Column must have name")
        self.exprs = []
        for item in self.values:
            item_expr = self.value_expr(item)
            self.exprs.append(item_expr)
        self.prepared = True
        return self

    def get_expr(self, current_schema: pl.Schema):
        for expr in self.exprs:
            if all(
                source in current_schema.names() for source in expr.meta.root_names()
            ):
                self.derived = True
                return {self.name: expr}
        return {}

