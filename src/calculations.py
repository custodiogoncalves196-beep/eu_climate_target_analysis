"""
Higher-level project calculations:
- reduction_%_atual (real % change vs base year)
- meta_2030 (theoretical target level based on base year)
- gap_2030 (how much remains to hit target)
"""
from __future__ import annotations

import pandas as pd


def add_reduction_percent_actual(
    df: pd.DataFrame,
    base_year: int = 1990,
    current_year: int = 2023,
    group_col: str = "country",
    year_col: str = "year",
    emissions_col: str = "emissions",
    out_col: str = "redução_%_atual",
) -> pd.DataFrame:
    """
    Computes the *actual* % change from base_year to current_year by group.
    Output repeated for all rows of that group (handy for filtering/reporting).
    reduction_%_atual = ((current - base) / base) * 100
    """
    out = df.copy()

    base = (out[out[year_col] == base_year]
            .groupby(group_col)[emissions_col]
            .sum()
            .rename("base"))
    curr = (out[out[year_col] == current_year]
            .groupby(group_col)[emissions_col]
            .sum()
            .rename("current"))

    kpi = pd.concat([base, curr], axis=1)
    kpi[out_col] = ((kpi["current"] - kpi["base"]) / kpi["base"]) * 100

    out = out.merge(kpi[[out_col]], left_on=group_col, right_index=True, how="left")
    return out


def add_target_2030_and_gap(
    df: pd.DataFrame,
    target_reduction_pct: float = -55.0,
    base_year: int = 1990,
    group_col: str = "country",
    year_col: str = "year",
    emissions_col: str = "emissions",
    out_target_col: str = "meta_2030",
    out_gap_col: str = "gap_2030",
    compare_year: int = 2023,
) -> pd.DataFrame:
    """
    meta_2030: target emissions level in 2030 based on base_year:
      meta_2030 = base * (1 + target_reduction_pct/100)
    gap_2030: how much needs to change from compare_year level to reach meta_2030:
      gap_2030 = meta_2030 - emissions(compare_year)
    """
    out = df.copy()

    base = (out[out[year_col] == base_year]
            .groupby(group_col)[emissions_col]
            .sum()
            .rename("base"))
    comp = (out[out[year_col] == compare_year]
            .groupby(group_col)[emissions_col]
            .sum()
            .rename("compare"))

    kpi = pd.concat([base, comp], axis=1)
    kpi[out_target_col] = kpi["base"] * (1 + target_reduction_pct / 100.0)
    kpi[out_gap_col] = kpi[out_target_col] - kpi["compare"]

    out = out.merge(kpi[[out_target_col, out_gap_col]], left_on=group_col, right_index=True, how="left")
    return out
