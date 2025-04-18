# %%
import json
import re

import pandas as pd
from tqdm import tqdm


urls = pd.read_csv("naep-parameter-urls.csv")

dfs = []
for _, subject, year, grade, scale, url in tqdm(
    urls.itertuples(), "Scraping parameters", total=len(urls)
):
    df = pd.read_html(url)[0]
    if "NAEP ID" not in df.columns:
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header

    def is_naep_id(string):
        return isinstance(string, str) and bool(re.match(r"^[A-Z][\dA-Z]+$", string))

    df = df[df["NAEP ID"].apply(is_naep_id)]
    assert len(df) > 0, f"No NAEP IDs found in {url}"

    if "aj" in df.columns:
        df = df.rename(
            columns={
                "aj": "a",
                "bj": "b",
                "cj": "c",
                "dj1": "d_1",
                "dj2": "d_2",
                "dj3": "d_3",
                "dj4": "d_4",
            }
        )
    df = df.rename(
        columns={
            "NAEP ID": "naep_id",
            "d": "d_1",
            "d.1": "d_2",
            "d.2": "d_3",
            "d.3": "d_4",
        }
    )
    df["item_id"] = df["naep_id"].apply(lambda naep_id: f"NAEP_{naep_id}")
    columns = ["item_id", "a", "b", "c"]
    for i in range(1, 5):
        if f"d_{i}" in df.columns:
            columns.append(f"d_{i}")
    df = df[columns]

    # Fix errors and convert to float
    for column in columns[1:]:
        df[column] = (
            df[column]
            .replace("#", 0)
            .replace("†", None)
            # Typo? https://nces.ed.gov/nationsreportcard/tdw/analysis/2000_2001/scaling_irt_math_2000_state_g8_geometry_acc.aspx
            .replace("6 0.587", 0.587)
            # Typo https://nces.ed.gov/nationsreportcard/tdw/analysis/2015/scaling_irt_g4sci_physical2015.aspx
            .replace("†0.37", 0.37)
            # Typo? https://nces.ed.gov/nationsreportcard/tdw/analysis/2000_2001/scaling_irt_hist_2001_natl_g4_democracy_acc.aspx
            .replace("(-1.266)", -1.266)
            .astype(float)
        )

    # Guessing parameters negative by mistake?
    # https://nces.ed.gov/nationsreportcard/tdw/analysis/2000_2001/scaling_irt_hist_2001_natl_g4_democracy_acc.aspx
    # https://nces.ed.gov/nationsreportcard/tdw/analysis/2000_2001/scaling_irt_hist_2001_natl_g8_technology_acc.aspx
    if year == 2001 and "subject" == "history" and scale == "demo" and grade == 4:
        df["c"] = -df["c"]
    if year == 2001 and "subject" == "history" and scale == "tech" and grade == 8:
        df["c"] = -df["c"]

    df["subject"] = subject
    df["year"] = year
    df["grade"] = grade
    df["scale"] = scale
    dfs.append(df)

parameters = pd.concat(dfs)
parameters = parameters.fillna("†").replace("†", None)

# %%
with open("output/naep-parameters.jsonl", "w") as f:
    for _, row in parameters.iterrows():
        row_dict = row.to_dict()
        if row_dict["d_1"] is None:
            row_dict["irt_model"] = "3pl"
            for i in range(1, 5):
                row_dict.pop(f"d_{i}")
            if row_dict["c"] is None:
                row_dict["c"] = 0.0
        else:
            row_dict["irt_model"] = "gpcm"
            for i in range(1, 5):
                if row_dict[f"d_{i}"] is None:
                    row_dict.pop(f"d_{i}")
            row_dict.pop("c")
        f.write(json.dumps(row_dict) + "\n")
