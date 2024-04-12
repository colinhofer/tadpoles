from typing import Any, List
from io import StringIO
from itertools import count
import polars as pl

ITER_MAX = 10

field = pl.col("__standin__")

def unnest_all(self: pl.DataFrame, seperator="."):
    def _unnest_all(struct_columns):
        return self.with_columns(
            [
                pl.col(column).struct.rename_fields(
                    [
                        f"{column}{seperator}{field_name}"
                        for field_name in self[column].struct.fields
                    ]
                )
                for column in struct_columns
            ]
        ).unnest(struct_columns)

    struct_columns = [name for name in self.columns if self[name].dtype == pl.Struct]
    while len(struct_columns):
        self = _unnest_all(struct_columns=struct_columns)
        struct_columns = [name for name in self.columns if self[name].dtype == pl.Struct]

    return self

pl.DataFrame.unnest_all = unnest_all

class Field(object):
    exprs: List[pl.Expr] = []
    
    def __init__(self, *inputs, **kwargs):
        self.exprs = []
        self.values = inputs
        self.primary_key = kwargs.get('primary_key', False)
        self.dtype = kwargs.get('dtype', str)
        self.default = kwargs.get('default')
        
    def __eq__(self, other):
        return self.name == other
    
    @property
    def literal(self) -> pl.Expr:
        return pl.lit(self.default).alias(self.name).cast(self.dtype)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name_str: str):
        self._name = name_str
        for value in self.values:
            if value is None:
                self.exprs.append(pl.col(self._name).cast(self.dtype))
            elif not isinstance(value, pl.Expr):
                self.default = value
                self.exprs.append(pl.col(self._name).cast(self.dtype).fill_null(self.default))
            else:
                self.exprs.append(pl.Expr.deserialize(
                    StringIO(value.meta.serialize().replace("__standin__", self._name))
                ).alias(self._name).cast(self.dtype))
    
        
    def derivable(self, columns: list):
        for expr in self.exprs:
            if all([source in columns for source in expr.meta.root_names()]):
                return expr


class PlModelMeta(type):
    def __new__(mcs, name: str, bases: Any, attrs: dict, **kwargs):
        fields: List[Field] = attrs.pop("__fields__", [])
        annotations: dict = attrs.get("__annotations__", {})
        model_attrs = {key: val for key, val in attrs.items() if not key.startswith("_") and not callable(val)}
        model_types = {key: val for key, val in annotations.items() if not key.startswith("_")}
        for name in set(model_attrs) | set(model_types):
            value = model_attrs.pop(name, None)
            dtype = model_types.get(name, type(value))
            field = value if isinstance(value, Field) else Field(value)
            field.dtype = dtype
            field.name = name
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
    
    
    def __init__(self, *args, derive: bool = True, unnest: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._derived = []
        if unnest:
            self._df = self.unnest_all()._df
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
