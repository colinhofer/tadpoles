from typing import Any, List, Literal
from io import StringIO
from itertools import count
import polars as pl

ITER_MAX = 10
NORM_LITS = Literal['explode', 'unnest', 'unnest-explode', 'unnest-first', 'explode-first']

field = pl.col("__standin__")

def model_starter(name: str, data: pl.DataFrame|dict):
    df = pl.DataFrame(data, infer_schema_length=None)
    print(f"class {name}(Model):")
    [print(f'   {col}: pl.{dtype} = pl.col("{col}")') for col, dtype in df.schema.items()]

def unnest_rename(df: pl.DataFrame, columns: list, separator: str = "."):
    return df.with_columns(
        [
            pl.col(column).struct.rename_fields(
                [
                    f"{column}{separator}{field_name}"
                    for field_name in df[column].struct.fields
                ]
            )
            for column in columns
        ]
    ).unnest(columns)   

def get_expandable(df: pl.DataFrame, how: str, columns: list = None):
    columns = columns or df.columns
    structs = [name for name in df.columns if df[name].dtype == pl.Struct and name in columns] if 'unnest' in how else []
    lists = [name for name in df.columns if df[name].dtype == pl.List and name in columns] if 'explode' in how else []
    return structs, lists

pl.DataFrame.get_expandable = get_expandable

def normalize(df: pl.DataFrame, separator: str = ".", columns: list = [], how: NORM_LITS = 'explode-unnest'):
    structs, lists = df.get_expandable(how, columns)
    while structs or lists:
        if lists:
            df = df.explode(lists)
        else:
            df = unnest_rename(df, structs, separator)
        if 'first' in how:
            return df
        structs, lists = df.get_expandable(how, columns)
    return df

pl.DataFrame.normalize = normalize


class Field(object):
    exprs: List[pl.Expr] = []
    name: str = None
    
    def __init__(self, *values, **kwargs):
        self.exprs = []
        self.values = values if values else (None,)
        self.primary_key = kwargs.get('primary_key', False)
        self.dtype = kwargs.get('dtype', str)
        self.default = kwargs.get('default')
        
    def __eq__(self, other):
        return self.name == other
    
    @property
    def literal(self) -> pl.Expr:
        return pl.lit(self.default).alias(self.name).cast(self.dtype)
    
    def set_exprs(self):
        if not self.name:
            raise ValueError(f"No name attribute for field")
        for value in self.values:
            if value is None:
                self.exprs.append(pl.col(self.name).cast(self.dtype))
            elif not isinstance(value, pl.Expr):
                self.default = value
                self.exprs.append(pl.col(self.name).cast(self.dtype).fill_null(self.default))
            else:
                self.exprs.append(pl.Expr.deserialize(
                    StringIO(value.meta.serialize().replace("__standin__", self.name))
                ).alias(self.name).cast(self.dtype))
    
    def derivable(self, context: list):
        for expr in self.exprs:
            if all([source in context for source in expr.meta.root_names()]):
                return expr


class PlModelMeta(type):
    
    def __new__(mcs, name: str, bases: Any, attrs: dict, **kwargs):
        fields: List[Field] = attrs.pop("__fields__", [])
        annotations: dict = attrs.get("__annotations__", {})
        model_attrs = {key: val for key, val in attrs.items() if not key.startswith("_") and not callable(val)}
        model_types = {key: val for key, val in annotations.items() if not key.startswith("_")}
        for col in set(model_attrs) | set(model_types):
            value = model_attrs.pop(col, None)
            dtype = model_types.get(col, type(value))
            field = value if isinstance(value, Field) else Field(value)
            field.dtype = dtype
            field.name = col
            field.set_exprs()
            fields.append(field)
        for base in bases:
            if hasattr(base, "__fields__"):
                fields.extend(base.__fields__)
        attrs["_key"] = [field.name for field in fields if field.primary_key]
        attrs["__fields__"] = fields
        return super().__new__(mcs, name, bases, attrs, **kwargs)


class Model(pl.DataFrame, metaclass=PlModelMeta):
    __iter_max__: int = ITER_MAX
    __tablename__: str
    __fields__: List[Field]
    _derived: List[Field]
    _key: List[str]
    
    def __init__(self, *args, derive: bool = True, normalize: NORM_LITS = None, normalize_columns: list = None, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except pl.exceptions.ComputeError:
            super().__init__(*args, infer_schema_length=None, **kwargs)
        self._derived = []
        self._df = self.normalize(how=normalize, columns=normalize_columns)._df if normalize else self._df
        self.new_cols = [field.name for field in self.__fields__]
        if derive:
            self._derive()

    @property
    def _underived(self):
        return [field for field in self.__fields__ if field not in self._derived]

    def _valid_exprs(self, lf: pl.LazyFrame):
        exprs = []
        for field in self._underived:
            expr = field.derivable(lf.columns)
            if expr is not None:
                exprs.append(expr)
                self._derived.append(field)
        return exprs
                            
    def _derive(self):
        if self.is_empty():
            self._df = pl.DataFrame(schema={field.name: field.dtype for field in self.__fields__})._df
            return
        lf = self.lazy()
        for n in count():
            if n > self.__iter_max__:
                raise RuntimeError(
                    f"Failed to derive columns {self._underived}. Exceeded maximum {self.__iter_max__} derivation iterations."
                )
            exprs = self._valid_exprs(lf)
            if not exprs:
                break
            lf = lf.with_columns(exprs)
        literals = [field.literal for field in self._underived]
        lf = lf.with_columns(literals).select(sorted(self.new_cols))
        self._df = lf.collect()._df     

    def with_columns(self, *exprs: Any, **named_exprs: Any):
        return self.__class__(super().with_columns(*exprs, **named_exprs), derive=False)

    def select(self, *exprs: Any, **named_exprs: Any):
        return self.__class__(super().select(*exprs, **named_exprs), derive=False)

    def update(self, *args, **kwargs):
        return self.__class__(super().update(*args, **kwargs), derive=False)
