from typing import List, Any, Literal
from itertools import count
from io import StringIO
import polars as pl


ITER_MAX = 10
NORM_LITS = Literal['explode', 'unnest', 'unnest-explode', 'unnest-first', 'explode-first']


field = pl.col("__standin__")

def model_starter(name: str, data: Any, expand: NORM_LITS = None):
    str_replace = [".", " ", "-", "/"]
    ldf = pl.LazyFrame(data, infer_schema_length=None)
    ldf = normalize(ldf, how=expand) if expand else ldf
    print(f"class {name}(Model):")
    for col, dtype in ldf.schema.items():
        field = col.lower()
        for char in str_replace:
            field = field.replace(char, "_")
        print(f'   {field}: pl.{dtype} = pl.col("{col}")')

def unnest_rename(ldf: pl.LazyFrame, columns: list, separator: str = "."):
    return ldf.with_columns(
        [
            pl.col(column).struct.rename_fields(
                [f"{column}{separator}{field.name}" for field in ldf.schema[column].fields]
            )
            for column in columns
        ]
    ).unnest(columns)   

def get_expandable(ldf: pl.LazyFrame, how: str, columns: list = None):
    columns = columns or ldf.columns
    structs = [name for name in ldf.columns if ldf.schema[name] == pl.Struct and name in columns] if 'unnest' in how else []
    lists = [name for name in ldf.columns if ldf.schema[name] == pl.List and name in columns] if 'explode' in how else []
    return structs, lists

def normalize(ldf: pl.LazyFrame, separator: str = ".", columns: list = [], how: NORM_LITS = 'explode-unnest') -> pl.DataFrame:
    structs, lists = get_expandable(ldf, how, columns)
    while structs or lists:
        if lists:
            ldf = ldf.explode(lists)
        else:
            ldf = unnest_rename(ldf, structs, separator)
        if 'first' in how:
            return ldf
        structs, lists = get_expandable(ldf, how, columns)
    return ldf

pl.LazyFrame.normalize = normalize


class Field(object):
    expressions: List[pl.Expr] = []
    name: str = None
    
    def __init__(self, *values, primary_key: bool = False, **kwargs):
        self.expressions = []
        self.values = values if values else (None,)
        self.primary_key = primary_key
        self.dtype = kwargs.get('dtype')
        self.default = kwargs.get('default')
        
    def __eq__(self, other):
        return self.name == other
    
    def __lt__(self, other):
        return self.name < other
    
    def __gt__(self, other):
        return self.name > other
    
    def __str__(self):
        return f"Field(name={self.name}, expressions={self.expressions}, dtype={self.dtype})"
    
    @property
    def literal(self) -> pl.Expr:
        return pl.lit(self.default).alias(self.name).cast(self.dtype)
    
    def set_exprs(self):
        if not self.name:
            raise ValueError(f"No name attribute for field")
        for value in self.values:
            if value is None:
                self.expressions.append(pl.col(self.name).cast(self.dtype))
            elif not isinstance(value, pl.Expr):
                self.default = value
                self.expressions.append(pl.col(self.name).cast(self.dtype).fill_null(self.default))
            else:
                self.expressions.append(pl.Expr.deserialize(
                    StringIO(value.meta.serialize().replace("__standin__", self.name))
                ).alias(self.name).cast(self.dtype))
    
    def derivable(self, context: list):
        for expr in self.expressions:
            if all([source in context for source in expr.meta.root_names()]):
                return expr


class TadpoleBase(pl.LazyFrame):
    __iter_max__: int = ITER_MAX
    primary_key: List[str]
    fields: List[Field]
    derived_fields: List[Field]
    
    def __init__(self, *args, derive: bool = True, expand: NORM_LITS = None, normalize_columns: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        if expand:
            self._ldf = self.normalize(how=expand, columns=normalize_columns)._ldf
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

    def underived_fields(self) -> list:
        return [field for field in self.fields if field not in self.derived_fields]
    
    def derive(self):
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
            
    def derivable_expressions(self) -> list:
        exprs = []
        for field in self.underived_fields():
            expr = field.derivable(self.columns)
            if expr is not None:
                exprs.append(expr)
                self.derived_fields.append(field)
        return exprs


class TadpoleMeta(type):
    
    def __new__(mcs, name: str, bases: Any, attrs: dict, db=None, **kwargs):
        fields: List[Field] = attrs.pop("__fields__", [])
        annotations: dict = attrs.get("__annotations__", {})
        model_attrs = {key: val for key, val in attrs.items() if not key.startswith("_") and not callable(val) and key not in TadpoleBase.__dict__}
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
    pass
