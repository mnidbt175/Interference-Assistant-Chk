"""Evaluate the assistant against ground truth computed independently in pandas.

Each question has a `truth` function written separately from assistant.py.
A question passes if the assistant's top label and value match the truth.
Run:  python eval.py
"""
import pandas as pd
from assistant import answer, load

df = load()
done = df[df["outcome"] == "escalated"]

# (question, truth) where truth is (label, value) of the top row, or a bare number
CASES = [
    ("How many cases are there in total?", len(df)),
    ("Which cause is most common?", df["cause"].value_counts().idxmax()),
    ("Which region has the most cases?", df["region"].value_counts().idxmax()),
    ("Which band has the most cases?", df["band"].value_counts().idxmax()),
    ("Which cause takes the longest to resolve on average?",
     df.groupby("cause")["days_to_close"].mean().idxmax()),
    ("What is the average days to close in the 900 MHz band?",
     round(df[df["band"] == "900 MHz"]["days_to_close"].mean(), 1)),
    ("How many cases involved jammers?", int((df["cause"] == "jammer").sum())),
    ("How many cases are in the North West region?", int((df["region"] == "North West").sum())),
    ("Which cause is most common in the South West?",
     df[df["region"] == "South West"]["cause"].value_counts().idxmax()),
    ("Which region is slowest to resolve cases on average?",
     df.groupby("region")["days_to_close"].mean().idxmax()),
    ("How many cases were escalated?", len(done)),
    ("What is the average days to close for illegal repeater cases?",
     round(df[df["cause"] == "illegal repeater"]["days_to_close"].mean(), 1)),
    ("Which year had the most cases?", df["year"].value_counts().idxmax()),
    ("Which band has the longest average resolution time?",
     df.groupby("band")["days_to_close"].mean().idxmax()),
    ("How many cases are in the 3500 MHz band?", int((df["band"] == "3500 MHz").sum())),
    # --- expanded schema; SINR means must skip cases with no measurement ---
    ("Which state has the most cases?", df["state"].value_counts().idxmax()),
    ("Which operator has the most cases?", df["operator"].value_counts().idxmax()),
    ("How many cases for Operator B?", int((df["operator"] == "Operator B").sum())),
    ("What is the average SINR for jammer cases?",
     round(df[df["cause"] == "jammer"]["sinr_db"].dropna().mean(), 1)),
    ("Which cause has the lowest average SINR?",
     df.groupby("cause")["sinr_db"].mean().idxmin()),
]


def passed(res: dict, truth) -> bool:
    if "rows" not in res:
        return False
    label, value = res["rows"][0]
    if isinstance(truth, str):
        return label == truth
    return abs(value - truth) < 0.05


if __name__ == "__main__":
    ok = 0
    for q, truth in CASES:
        res = answer(q, df)
        good = passed(res, truth)
        ok += good
        print(("PASS " if good else "FAIL ") + q)
        if not good:
            print("     expected:", truth, "| got:", res.get("rows", res)[:2])
    print(f"\n{ok}/{len(CASES)} correct ({ok / len(CASES):.0%})")
