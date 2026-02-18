"""
Merging utilities:
- emissions with population, typically on (country, year)
"""
from __future__ import annotations

import pandas as pd
from typing import Literal


def population_wide_to_long(
    df_population_wide: pd.DataFrame,
    country_col: str = "Country Name",
    year_start: int = 1990,
    year_end: int = 2023,
    out_country_col: str = "country",
    out_year_col: str = "year",
    out_value_col: str = "population",
) -> pd.DataFrame:
    """
    Convert population wide table (Country Name + years) to long (country, year, population).
    """
    df = df_population_wide.copy()
    years = [str(y) for y in range(year_start, year_end + 1)]
    year_cols = [c for c in years if c in df.columns]
    long_df = df.melt(
        id_vars=[country_col],
        value_vars=year_cols,
        var_name=out_year_col,
        value_name=out_value_col,
    )
    long_df[out_year_col] = pd.to_numeric(long_df[out_year_col], errors="coerce").astype("Int64")
    long_df[out_value_col] = pd.to_numeric(long_df[out_value_col], errors="coerce").astype("Int64")
    long_df = long_df.rename(columns={country_col: out_country_col})
    return long_df


def merge_emissions_population_long(
    emissions_long: pd.DataFrame,
    population_long: pd.DataFrame,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    country_col: str = "country",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Merge long-format emissions with long-format population.
    """
    em = emissions_long.copy()
    pop = population_long.copy()

    if country_col not in em.columns or year_col not in em.columns:
        raise ValueError(f"Emissions must contain '{country_col}' and '{year_col}'.")
    if country_col not in pop.columns or year_col not in pop.columns:
        raise ValueError(f"Population must contain '{country_col}' and '{year_col}'.")

    merged = em.merge(pop, on=[country_col, year_col], how=how)
    return merged
