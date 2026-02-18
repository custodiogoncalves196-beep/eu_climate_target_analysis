import pandas as pd
from src.data_cleaning import clean_emissions_raw, clean_population_raw
from src.feature_engineering import add_total_emissions, add_difference_between_years, wide_to_long_years, add_sector_main
from src.merge_data import population_wide_to_long, merge_emissions_population_long
from src.calculations import add_reduction_percent_actual, add_target_2030_and_gap
from src.feature_engineering import add_emissions_per_capita, apply_rounding

# 1) Load raw files
em_raw = pd.read_csv("data/raw/estat_env_air_gge.tsv", sep="\t")
pop_raw = pd.read_csv("data/raw/populacao.csv", skiprows=4)

# 2) Clean
em = clean_emissions_raw(em_raw)
pop = clean_population_raw(pop_raw)

# 3) Features on emissions (wide)
em = add_total_emissions(em)
em = add_difference_between_years(em, 1990, 2023)

# 4) Long format
id_cols = ["freq","unit","airpol","src_crf","geo","country"]
em_long = wide_to_long_years(em, id_cols=id_cols)
em_long = add_sector_main(em_long)

pop_long = population_wide_to_long(pop)

# 5) Merge + per capita
df = merge_emissions_population_long(em_long, pop_long, how="inner")
df = add_emissions_per_capita(df, emissions_col="emissions", population_col="population")

# 6) Targets & KPIs (optional)
df = add_reduction_percent_actual(df, base_year=1990, current_year=2023)
df = add_target_2030_and_gap(df, target_reduction_pct=-55.0, base_year=1990, compare_year=2023)

# 7) Rounding / export
df = apply_rounding(df, round_5_cols=["emissions","emissions_per_capita","gap_2030","meta_2030"], percent_cols=["redução_%_atual"])
df.to_csv("data/processed/merged_dataset.tsv", sep="\t", index=False)

print("Saved: data/processed/merged_dataset.tsv")
