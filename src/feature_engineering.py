"""
Feature engineering:
- totals, differences, per-capita metrics
- wide<->long transformations
- sector mapping
"""
from __future__ import annotations

import pandas as pd
from typing import Iterable, Optional


def add_total_emissions(df: pd.DataFrame, year_start: int = 1990, year_end: int = 2023,
                       out_col: str = "total_emissoes") -> pd.DataFrame:
    """
    Sum emissions across year columns (wide table) to create `total_emissoes`,
    mirroring what you did in the notebook.
    """
    out = df.copy()
    years = [str(y) for y in range(year_start, year_end + 1)]
    available = [c for c in years if c in out.columns]
    if not available:
        raise ValueError("No year columns found to compute total emissions.")

    out[available] = out[available].apply(pd.to_numeric, errors="coerce")
    out[out_col] = out[available].sum(axis=1)
    return out


def add_difference_between_years(df: pd.DataFrame, year_a: int = 1990, year_b: int = 2023,
                                out_col: str = "diferença_total") -> pd.DataFrame:
    """
    Create a difference column (year_b - year_a) like your `diferença_total`.
    """
    out = df.copy()
    a, b = str(year_a), str(year_b)
    if a not in out.columns or b not in out.columns:
        raise ValueError(f"Expected year columns {a} and {b}.")

    out[a] = pd.to_numeric(out[a], errors="coerce")
    out[b] = pd.to_numeric(out[b], errors="coerce")
    out = out.dropna(subset=[a, b]).copy()
    out[out_col] = out[b] - out[a]
    return out


def wide_to_long_years(
    df: pd.DataFrame,
    id_cols: Iterable[str],
    year_start: int = 1990,
    year_end: int = 2023,
    value_name: str = "emissions",
    var_name: str = "year",
) -> pd.DataFrame:
    """
    Convert wide year columns (1990..2023) into long format.
    Produces: id_cols + [year, emissions]
    """
    years = [str(y) for y in range(year_start, year_end + 1)]
    year_cols = [c for c in years if c in df.columns]

    out = df.copy()
    out_long = out.melt(
        id_vars=list(id_cols),
        value_vars=year_cols,
        var_name=var_name,
        value_name=value_name,
    )
    out_long[var_name] = pd.to_numeric(out_long[var_name], errors="coerce").astype("Int64")
    out_long[value_name] = pd.to_numeric(out_long[value_name], errors="coerce")
    return out_long


def map_sector_main(code: str) -> str:
    """
    Same sector mapping logic you used (CRF1..CRF6, TOTX memo).
    """
    if not isinstance(code, str):
        return "Outro"
    if code.startswith("CRF1"):
        return "Energia"
    if code.startswith("CRF2"):
        return "Processos Industriais"
    if code.startswith("CRF3"):
        return "Agricultura"
    if code.startswith("CRF4"):
        return "Uso do Solo e Florestas (LULUCF)"
    if code.startswith("CRF5"):
        return "Resíduos"
    if code.startswith("CRF6"):
        return "Outros"
    if code.startswith("TOTX"):
        return "Memo"
    return "Outro"


def add_sector_main(df: pd.DataFrame, src_col: str = "src_crf", out_col: str = "sector_main") -> pd.DataFrame:
    out = df.copy()
    if src_col not in out.columns:
        raise ValueError(f"Expected column '{src_col}'.")
    out[out_col] = out[src_col].apply(map_sector_main)
    return out


def add_emissions_per_capita(
    df: pd.DataFrame,
    emissions_col: str = "emissions",
    population_col: str = "population",
    out_col: str = "emissions_per_capita",
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out[emissions_col] / out[population_col]
    return out


def apply_rounding(
    df: pd.DataFrame,
    round_5_cols: Optional[Iterable[str]] = None,
    percent_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to match your formatting requirement:
    - some columns rounded to 5 decimals
    - percentage columns rounded to 0 decimals
    """
    out = df.copy()
    if round_5_cols:
        for c in round_5_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(5)

    if percent_cols:
        for c in percent_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(0)

    return out
