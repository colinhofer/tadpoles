import polars as pl
from typing import List, Literal, Dict, Tuple, Any
from itertools import count
from .field import Field


ITER_MAX = 10
NORM_LITS = Literal[
    "explode", "unnest", "unnest-explode", "unnest-first", "explode-first"
]


def model_starter(name: str, data: Any, expand: NORM_LITS = None) -> None:
    """Prints a basic Tadpoles model class with fields based on the provided data schema."""
    str_replace = [".", " ", "-", "/"]
    ldf = pl.LazyFrame(data, infer_schema_length=None)
    ldf = normalize(ldf, how=expand) if expand else ldf
    print(f"class {name}(Model):")
    for col, dtype in ldf.collect_schema().items():
        field = col.lower()
        for char in str_replace:
            field = field.replace(char, "_")
        print(f'   {field}: pl.{dtype} = pl.col("{col}")')


def get_expandable(
    how: str, schema: Dict[str, pl.DataType], columns: List[str] = None
) -> Tuple[List[str], List[str]]:
    """Identifies expandable columns (structs and lists) based on the provided method."""
    if columns:
        columns = [col for col in schema if any(name in col for name in columns)]
    else:
        columns = list(schema.keys())

    structs = (
        [name for name in columns if isinstance(schema[name], pl.Struct)]
        if "unnest" in how
        else []
    )
    lists = (
        [name for name in columns if isinstance(schema[name], pl.List)]
        if "explode" in how
        else []
    )

    return structs, lists


def unnest_rename(
    ldf: pl.LazyFrame,
    columns: List[str],
    schema: Dict[str, pl.DataType],
    separator: str = ".",
) -> pl.LazyFrame:
    """Unnests and renames columns in the LazyFrame."""
    return ldf.with_columns(
        [
            pl.col(column).struct.rename_fields(
                [f"{column}{separator}{field.name}" for field in schema[column].fields]
            )
            for column in columns
        ]
    ).unnest(columns)


def normalize(
    ldf: pl.LazyFrame,
    separator: str = ".",
    columns: List[str] = [],
    how: NORM_LITS = "explode-unnest",
) -> pl.LazyFrame:
    """Normalizes the LazyFrame by expanding and unnesting columns."""
    if not how:
        return ldf

    schema = ldf.collect_schema()
    structs, lists = get_expandable(how, schema, columns)

    while structs or lists:
        if lists:
            ldf = ldf.explode(lists)
        else:
            ldf = unnest_rename(ldf, structs, schema, separator)

        if "first" in how:
            break

        schema = ldf.collect_schema()
        structs, lists = get_expandable(how, schema, columns)

    return ldf


def get_exprs(columns: List[Field], current_schema: pl.Schema):
    exprs = {}
    for col in columns:
        if not col.derived:
            exprs.update(col.get_expr(current_schema))
    return exprs


def transform(ldf: pl.LazyFrame, fields: List[Field], max_iterations: int = ITER_MAX):
    [setattr(col, 'derived', False) for col in fields]
    lf_schema = ldf.collect_schema()
    for n in count():
        if n > max_iterations:
            raise RuntimeError(
                f"""
                Failed to derive fields {[col for col in fields if not col.derived]}.
                Exceeded maximum {max_iterations} derivation iterations.
                """
            )
        exprs = get_exprs(fields, lf_schema)
        if not exprs:
            break
        ldf = ldf.with_columns(**exprs)
        lf_schema.update({name: pl.DataType for name in exprs.keys()})
    literals = {col.name: col.literal for col in fields if not col.derived}
    ldf = ldf.with_columns(**literals).select(sorted([col.name for col in fields]))
    return ldf

def append(lf: pl.LazyFrame, other: pl.LazyFrame):
    other_schema = other.collect_schema()
    current_schema = lf.collect_schema()
    all_columns = set(current_schema.names()) | set(other_schema.names())
    lf1 = lf.with_columns([pl.lit(None).cast(other_schema.get(col)).alias(col) for col in all_columns if col not in current_schema.names()])
    lf2 = other.with_columns([pl.lit(None).cast(current_schema.get(col)).alias(col) for col in all_columns if col not in other_schema.names()])
    return pl.concat([
        lf1.select(all_columns),
        lf2.select(all_columns)
    ])
