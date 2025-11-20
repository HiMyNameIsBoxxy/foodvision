import pandas as pd

NUTRIENT_CSV = "data/usda/nutrient.csv"

df = pd.read_csv(NUTRIENT_CSV)

# Look for Energy in kcal
print(df[df["name"].str.contains("Energy", case=False, na=False)][[
    "id", "name", "unit_name"
]])
