"""
Data cleaning functions for the EU climate target project.

Goal: keep 'raw' files untouched, and create consistent cleaned DataFrames
that downstream steps can rely on.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional


DEFAULT_COUNTRY_MAP: Dict[str, str] = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "NO": "Norway",
    "IS": "Iceland",
    "EU27_2020": "European Union (EU27)"
}

DEFAULT_EU_COUNTRIES: List[str] = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Cyprus",
    "Czech Republic",
    "Germany",
    "Denmark",
    "Estonia",
    "Greece",
    "Spain",
    "Finland",
    "France",
    "Croatia",
    "Hungary",
    "Ireland",
    "Italy",
    "Lithuania",
    "Luxembourg",
    "Latvia",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Sweden",
    "Slovenia",
    "Slovakia",
    "Iceland",
    "Norway"
]


def clean_emissions_raw(
    df_raw: pd.DataFrame,
    country_map: Dict[str, str] = DEFAULT_COUNTRY_MAP,
    drop_eu_total: bool = True,
) -> pd.DataFrame:
    """
    Clean the Eurostat emissions TSV (wide by year) as you did in the notebook:
    - split the combined key column into: freq, unit, airpol, src_crf, geo
    - strip column names
    - map geo code -> country name
    - (optional) remove EU27_2020 aggregate rows
    """
    df = df_raw.copy()

    key_col = "freq,unit,airpol,src_crf,geo\TIME_PERIOD"
    if key_col not in df.columns:
        raise ValueError(f"Expected column '{key_col}' in emissions raw dataframe.")

    df.columns = df.columns.str.strip()

    # Split the key column
    parts = df[key_col].astype(str).str.split(",", expand=True)
    parts.columns = ["freq", "unit", "airpol", "src_crf", "geo"]
    df = pd.concat([df.drop(columns=[key_col]), parts], axis=1)

    # Map country
    df["country"] = df["geo"].map(country_map)

    if drop_eu_total:
        df = df[df["geo"] != "EU27_2020"].copy()

    return df


def clean_population_raw(
    df_raw: pd.DataFrame,
    eu_countries: List[str] = DEFAULT_EU_COUNTRIES,
) -> pd.DataFrame:
    """
    Clean the World Bank-style population file (wide by year) as in your notebook:
    - keep only EU-related countries list
    - drop metadata columns
    - keep years 1990..2023 as strings
    - convert to numeric Int64
    Output columns: Country Name + year columns
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # Keep only EU countries
    if "Country Name" not in df.columns:
        raise ValueError("Expected column 'Country Name' in population dataframe.")

    df = df[df["Country Name"].isin(eu_countries)].copy()

    # Drop metadata if present
    for col in ["Country Code", "Indicator Name", "Indicator Code"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    years = [str(y) for y in range(1990, 2024)]
    keep_cols = ["Country Name"] + [c for c in years if c in df.columns]
    df = df[keep_cols].copy()

    # Convert year columns to Int64
    year_cols = [c for c in df.columns if c.isdigit()]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    return df
