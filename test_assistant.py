import pytest
from assistant import answer, load, parse_rules, run_query, validate

df = load()


def test_total_count_matches_dataset():
    assert answer("How many cases are there in total?", df)["rows"][0][1] == len(df)


def test_group_by_cause_sorted_descending():
    values = [v for _, v in answer("Which cause is most common?", df)["rows"]]
    assert values == sorted(values, reverse=True)


def test_filter_reduces_cases():
    res = answer("How many cases are in the 900 MHz band?", df)
    assert res["rows"][0][1] == int((df["band"] == "900 MHz").sum())


def test_unanswerable_question_is_refused():
    assert "error" in answer("What is the weather in Abuja?", df)


def test_unknown_group_field_is_refused():
    assert "error" in answer("Which technician has the most cases?", df)


def test_validate_rejects_invented_column():
    bad = {"metric": "count", "group_by": "technician", "filters": {}}
    with pytest.raises(ValueError):
        validate(bad, df)


def test_validate_rejects_invented_value():
    bad = {"metric": "count", "group_by": None, "filters": {"band": "9999 MHz"}}
    with pytest.raises(ValueError):
        validate(bad, df)


def test_run_query_ascending_flag():
    spec = validate({"metric": "mean_days", "group_by": "cause", "filters": {},
                     "ascending": True}, df)
    values = [v for _, v in run_query(df, spec)["rows"]]
    assert values == sorted(values)


def test_sinr_mean_skips_unmeasured_cases():
    res = answer("What is the average SINR for jammer cases?", df)
    jam = df[df["cause"] == "jammer"]
    assert res["n_used"] == int(jam["sinr_db"].notna().sum())
    assert res["n_used"] < len(jam)  # some jammer cases have no measurement
    assert res["rows"][0][1] == round(jam["sinr_db"].mean(), 1)


def test_unmeasured_cases_have_no_rf_values():
    rf = ["measured_power_dbm", "noise_floor_dbm", "sinr_db", "occupied_bandwidth_mhz"]
    missing = df[df["sinr_db"].isna()]
    assert len(missing) > 0 and missing[rf].isna().all().all()


def test_measurement_samples_link_to_cases():
    import pandas as pd
    from assistant import DATA_PATH
    m = pd.read_csv(DATA_PATH.with_name("measurements.csv"))
    assert set(m["case_id"]) <= set(df["case_id"])
    assert set(m["case_id"]) == set(df.loc[df["sinr_db"].notna(), "case_id"])


def test_slowest_is_not_read_as_lowest():
    spec = parse_rules("Which region is slowest to resolve cases on average?", df)
    assert spec["ascending"] is False
