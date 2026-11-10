"""Generate a synthetic interference-case dataset (cases, measurements, equipment).

All records are FAKE. No NCC or operator data is used. Operator names are
placeholders, coordinates are approximate state centres plus random jitter,
and patterns (some causes take longer, jammers have poor SINR) are invented.

RF measurement fields (power, noise floor, SINR, occupied bandwidth) are only
filled for ~60% of cases, as in real life, where not every case is measured.
"""
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
N_CASES = 300
P_MEASURED = 0.6

# region -> {state: (approx lat, approx lon)}
STATES = {
    "North Central": {"FCT": (9.06, 7.49), "Kwara": (8.50, 4.55), "Plateau": (9.90, 8.90)},
    "North East": {"Borno": (11.80, 13.15), "Adamawa": (9.30, 12.40), "Bauchi": (10.30, 9.80)},
    "North West": {"Kano": (12.00, 8.50), "Kaduna": (10.50, 7.40), "Sokoto": (13.06, 5.24)},
    "South East": {"Enugu": (6.45, 7.50), "Anambra": (6.20, 6.90), "Imo": (5.50, 7.00)},
    "South South": {"Rivers": (4.80, 6.99), "Delta": (5.50, 5.80), "Akwa Ibom": (5.00, 7.85)},
    "South West": {"Lagos": (6.52, 3.38), "Oyo": (7.85, 3.93), "Ogun": (7.00, 3.35)},
}
# band -> (low MHz, high MHz) used to pick a plausible frequency
BANDS = {
    "700 MHz": (703, 748), "800 MHz": (791, 821), "900 MHz": (880, 915),
    "1800 MHz": (1710, 1785), "2100 MHz": (1920, 1980),
    "2600 MHz": (2500, 2570), "3500 MHz": (3400, 3600),
}
# cause -> (relative frequency, typical days to close, typical SINR dB, source equipment)
CAUSES = {
    "illegal repeater": (25, 6, 0, "signal booster"),
    "faulty equipment": (20, 9, 8, "base station transmitter"),
    "unlicensed transmitter": (18, 14, 3, "unlicensed radio"),
    "intermodulation": (12, 11, 4, "co-sited transmitters"),
    "jammer": (10, 4, -8, "signal jammer"),
    "cross-border signal": (8, 30, 2, "foreign broadcast/base station"),
    "power line noise": (7, 18, 5, "damaged power line"),
}
NO_LICENCE = {"illegal repeater", "unlicensed transmitter", "jammer", "cross-border signal"}
OPERATORS = ["Operator A", "Operator B", "Operator C", "Operator D"]
OUTCOMES = ["resolved", "escalated", "referred"]
EQUIPMENT = [
    ("MON-01", "fixed monitoring station", "2023-03-01"),
    ("MON-02", "fixed monitoring station", "2023-09-15"),
    ("MON-03", "mobile monitoring unit", "2024-01-20"),
    ("MON-04", "mobile monitoring unit", "2024-06-10"),
    ("MON-05", "handheld spectrum analyser", "2024-02-05"),
    ("MON-06", "handheld spectrum analyser", "2025-01-12"),
]


def build(seed: int = SEED, n: int = N_CASES):
    rng = random.Random(seed)
    causes = list(CAUSES)
    weights = [CAUSES[c][0] for c in causes]
    start = date(2023, 1, 1)
    cases, samples = [], []
    for i in range(1, n + 1):
        cause = rng.choices(causes, weights)[0]
        _, typical_days, typical_sinr, source = CAUSES[cause]
        region = rng.choice(list(STATES))
        state = rng.choice(list(STATES[region]))
        lat0, lon0 = STATES[region][state]
        band = rng.choice(list(BANDS))
        low, high = BANDS[band]
        freq = round(rng.uniform(low, high), 1)
        licensed = cause not in NO_LICENCE
        reported = start + timedelta(days=rng.randint(0, 3 * 365 - 1))
        # suspected type is what staff first thought; it is sometimes wrong
        suspected = cause if rng.random() < 0.85 else rng.choice(causes)

        row = {
            "case_id": f"SYN-{i:04d}", "date_reported": reported,
            "region": region, "state": state,
            "location": f"{state} site {rng.randint(1, 40)}",
            "latitude": round(lat0 + rng.uniform(-0.4, 0.4), 4),
            "longitude": round(lon0 + rng.uniform(-0.4, 0.4), 4),
            "operator": rng.choice(OPERATORS) if licensed else "Unidentified",
            "band": band, "frequency_mhz": freq,
            "bandwidth_mhz": rng.choice([5, 10, 15, 20]) if licensed else None,
            "measured_power_dbm": None, "noise_floor_dbm": None,
            "sinr_db": None, "occupied_bandwidth_mhz": None,
            "suspected_interference_type": suspected, "cause": cause,
            "source_equipment": source,
            "monitoring_equipment": rng.choice(EQUIPMENT)[0],
            "assigned_frequency_mhz": round(freq / 0.2) * 0.2 if licensed else None,
            "licence_power_limit_dbm": 43 if licensed else None,
            "days_to_close": max(1, round(rng.gauss(typical_days, typical_days * 0.35))),
            "outcome": rng.choices(OUTCOMES, [80, 12, 8])[0],
        }
        if rng.random() < P_MEASURED:  # RF fields only where a measurement exists
            power = round(rng.gauss(-85, 6), 1)
            floor = round(rng.gauss(-110, 2), 1)
            row.update(
                measured_power_dbm=power, noise_floor_dbm=floor,
                sinr_db=round(rng.gauss(typical_sinr, 3), 1),
                occupied_bandwidth_mhz=round(rng.uniform(0.2, 20), 1),
            )
            t0 = datetime.combine(reported, datetime.min.time()) + timedelta(hours=rng.randint(6, 18))
            for k in range(5):  # timestamped samples taken during the visit
                samples.append({
                    "case_id": row["case_id"],
                    "timestamp": (t0 + timedelta(minutes=10 * k)).isoformat(),
                    "frequency_mhz": freq,
                    "power_dbm": round(power + rng.gauss(0, 1.5), 1),
                    "noise_floor_dbm": round(floor + rng.gauss(0, 0.8), 1),
                })
        cases.append(row)
    equipment = pd.DataFrame(EQUIPMENT, columns=["equipment_id", "type", "last_calibration"])
    return pd.DataFrame(cases), pd.DataFrame(samples), equipment


if __name__ == "__main__":
    out = Path(__file__).with_name("data")
    out.mkdir(exist_ok=True)
    cases, samples, equipment = build()
    cases.to_csv(out / "cases.csv", index=False)
    samples.to_csv(out / "measurements.csv", index=False)
    equipment.to_csv(out / "equipment.csv", index=False)
    print(f"cases: {len(cases)} ({cases['sinr_db'].notna().sum()} with RF measurements)")
    print(f"measurement samples: {len(samples)} | equipment: {len(equipment)}")
