"""Interference-case assistant.

Pipeline (the LLM never touches the numbers):
  1. PARSE   question -> a small JSON "query spec" (LLM if an API key is set,
             otherwise a rule-based parser).
  2. VALIDATE the spec against a whitelist, so a model can't invent columns
             or values.
  3. RUN     the spec with pandas on the case data.
  4. EXPLAIN the result in plain language (LLM or a template).

Because step 3 is ordinary code, every number in an answer can be checked.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).with_name("data") / "cases.csv"
MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6")

GROUPABLE = ["cause", "band", "region", "state", "operator", "year"]
FILTERABLE = ["cause", "band", "region", "state", "operator", "outcome"]
METRICS = ["count", "mean_days", "mean_sinr"]


def load(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_reported"])
    df["year"] = df["date_reported"].dt.year.astype(str)
    return df


def allowed_values(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].astype(str).unique()) for c in set(GROUPABLE + FILTERABLE)}


# ---------- 1. PARSE ----------
def parse_rules(question: str, df: pd.DataFrame) -> dict:
    """Rule-based parser. Works offline; deliberately simple."""
    q = question.lower()
    spec = {"metric": "count", "group_by": None, "filters": {}, "ascending": False}

    time_words = ("average", "longest", "slowest", "fastest", "quickest",
                  "shortest", "to resolve", "resolution", "days", "take")
    if any(w in q for w in time_words):
        spec["metric"] = "mean_days"
    if re.search(r"sinr|signal[- ]to[- ]noise|signal quality", q):
        spec["metric"] = "mean_sinr"
    if re.search(r"\b(fastest|quickest|shortest|lowest|worst|weakest)\b", q):
        spec["ascending"] = True  # whole words only: "slowest" must not match "lowest"

    m = re.search(r"which (cause|band|region|state|operator|year)", q)
    if m:
        spec["group_by"] = m.group(1)

    for region in allowed_values(df)["region"]:
        if region.lower() in q:
            spec["filters"]["region"] = region
    for state in allowed_values(df)["state"]:
        if re.search(rf"\b{re.escape(state.lower())}\b", q):
            spec["filters"]["state"] = state
    m_op = re.search(r"operator ([a-d])\b", q)
    if m_op and f"Operator {m_op.group(1).upper()}" in allowed_values(df)["operator"]:
        spec["filters"]["operator"] = f"Operator {m_op.group(1).upper()}"
    for mhz in re.findall(r"(\d{3,4})\s*mhz", q):
        band = f"{mhz} MHz"
        if band in allowed_values(df)["band"]:
            spec["filters"]["band"] = band
    cause_keys = {"jammer": "jammer", "repeater": "illegal repeater",
                  "faulty": "faulty equipment", "unlicensed": "unlicensed transmitter",
                  "intermodulation": "intermodulation",
                  "cross-border": "cross-border signal", "cross border": "cross-border signal",
                  "power line": "power line noise"}
    for key, cause in cause_keys.items():
        if key in q:
            spec["filters"]["cause"] = cause
    for outcome in ("escalated", "resolved", "referred"):
        if outcome in q:
            spec["filters"]["outcome"] = outcome
    # A filter on the same field as the grouping would be circular; drop it.
    spec["filters"].pop(spec["group_by"], None)

    # Refuse questions we can't ground in the data instead of guessing.
    asks_unknown = re.search(r"which (?!cause|band|region|state|operator|year)\w+", q)
    has_signal = (spec["group_by"] or spec["filters"]
                  or re.search(r"how many|total|number of|average", q))
    if asks_unknown or not has_signal:
        return {"error": "unanswerable"}
    return spec


PARSE_PROMPT = """You convert questions about radio-interference cases into JSON.
Return ONLY a JSON object, no prose, with keys:
  "metric": one of {metrics}
  "group_by": null or one of {groupable}
  "filters": object whose keys are from {filterable}; values must come from: {values}
  "ascending": true only if the question asks for the smallest/fastest/shortest.
If the question cannot be answered from this data, return {{"error": "unanswerable"}}."""


def parse_llm(question: str, df: pd.DataFrame) -> dict:
    import anthropic  # imported lazily so offline mode needs no SDK
    client = anthropic.Anthropic()
    system = PARSE_PROMPT.format(metrics=METRICS, groupable=GROUPABLE,
                                 filterable=FILTERABLE, values=json.dumps(allowed_values(df)))
    msg = client.messages.create(model=MODEL, max_tokens=300, system=system,
                                 messages=[{"role": "user", "content": question}])
    text = msg.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


# ---------- 2. VALIDATE ----------
def validate(spec: dict, df: pd.DataFrame) -> dict:
    """Raise ValueError unless the spec only uses known fields and values."""
    if "error" in spec:
        raise ValueError(spec["error"])
    if spec.get("metric") not in METRICS:
        raise ValueError(f"bad metric: {spec.get('metric')}")
    if spec.get("group_by") not in [None] + GROUPABLE:
        raise ValueError(f"bad group_by: {spec.get('group_by')}")
    vals = allowed_values(df)
    for k, v in spec.get("filters", {}).items():
        if k not in FILTERABLE or v not in vals[k]:
            raise ValueError(f"bad filter: {k}={v}")
    return {"metric": spec["metric"], "group_by": spec.get("group_by"),
            "filters": spec.get("filters", {}), "ascending": bool(spec.get("ascending"))}


# ---------- 3. RUN ----------
def run_query(df: pd.DataFrame, spec: dict) -> dict:
    sub = df
    for k, v in spec["filters"].items():
        sub = sub[sub[k] == v]
    n = len(sub)
    col = {"mean_days": "days_to_close", "mean_sinr": "sinr_db"}.get(spec["metric"])
    # RF fields are only filled where a measurement exists; means skip the blanks
    n_used = int(sub[col].notna().sum()) if col else n
    if spec["group_by"] is None:
        value = n if col is None else round(float(sub[col].mean()), 1)
        return {"n_cases": n, "n_used": n_used, "rows": [("all matching cases", value)]}
    g = sub.groupby(spec["group_by"])
    series = g.size() if col is None else g[col].mean().dropna()
    series = series.sort_values(ascending=spec["ascending"])  # rank first, round after
    if col is not None:
        series = series.round(1)
    return {"n_cases": n, "n_used": n_used,
            "rows": [(str(k), float(v)) for k, v in series.items()]}


# ---------- 4. EXPLAIN ----------
def explain(question: str, spec: dict, result: dict, use_llm: bool) -> str:
    unit = {"count": "cases", "mean_days": "days to close (average)",
            "mean_sinr": "SINR in dB (average)"}[spec["metric"]]
    note = ""
    if spec["metric"] == "mean_sinr":
        note = f" Based on {result['n_used']} of {result['n_cases']} cases that have SINR measurements."
    scope = ", ".join(f"{k}={v}" for k, v in spec["filters"].items()) or "all cases"
    rows = result["rows"]
    if use_llm:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (f"Question: {question}\nScope: {scope}\nMetric: {unit}\n"
                  f"Results (label, value): {rows[:8]}\nCases in scope: {result['n_cases']}\n"
                  "Answer in 2-3 plain sentences using ONLY these numbers. "
                  "Mention that the data is synthetic." + (note and f" Also say:{note}"))
        msg = client.messages.create(model=MODEL, max_tokens=250,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    if spec["group_by"] is None:
        return f"{rows[0][1]} {unit} for {scope} (synthetic data).{note}"
    top = ", ".join(f"{k}: {v:g}" for k, v in rows[:3])
    return f"By {spec['group_by']} ({unit}), top results: {top}. Scope: {scope} (synthetic data).{note}"


# ---------- public entry point ----------
def answer(question: str, df: pd.DataFrame = None) -> dict:
    df = load() if df is None else df
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        raw = parse_llm(question, df) if use_llm else parse_rules(question, df)
        spec = validate(raw, df)
    except Exception as exc:  # fall back to rules, never run an unvalidated spec
        if use_llm:
            try:
                spec = validate(parse_rules(question, df), df)
                use_llm = False
            except Exception:
                return {"error": str(exc), "text": "Sorry, I can't answer that from this data."}
        else:
            return {"error": str(exc), "text": "Sorry, I can't answer that from this data."}
    result = run_query(df, spec)
    return {"spec": spec, "rows": result["rows"], "n_cases": result["n_cases"], "n_used": result["n_used"],
            "text": explain(question, spec, result, use_llm)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:]))["text"])
    else:
        print("Ask about the synthetic cases (blank line to quit).")
        while (q := input("> ").strip()):
            print(answer(q)["text"])
