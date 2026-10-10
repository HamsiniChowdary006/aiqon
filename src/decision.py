"""Turn a congestion probability into an operational decision.

Bands (from the prep guide):
  > 70%   HIGH   -> act now
  40-70%  MEDIUM -> monitor, keep buffer
  < 40%   LOW    -> proceed normally

Run:  python src/decision.py     (shows how many test shipments fall in each band)
"""
from pathlib import Path
import joblib
import pandas as pd

HIGH, LOW = 0.70, 0.40

ACTIONS = {
    "HIGH": ["Reroute to an alternate port or slow-steam to delay arrival",
             "Alert the consignee and customs broker about a likely delay",
             "Rebook trucking / rail slots to a later window"],
    "MEDIUM": ["Monitor queue and weather updates every few hours",
               "Keep buffer time in the downstream schedule",
               "Pre-clear documents so berthing is not delayed further"],
    "LOW": ["Proceed with the normal plan"],
}


def risk_band(p):
    return "HIGH" if p > HIGH else ("MEDIUM" if p >= LOW else "LOW")


def recommend(p, confidence="High"):
    band = risk_band(p)
    actions = list(ACTIONS[band])
    if confidence == "Low" and band != "LOW":
        actions.insert(0, "Confidence is low: verify live queue and weather data before acting")
    return {"band": band, "actions": actions}


def band_summary(y_true, p):
    """How many shipments land in each band, and how often they were really congested."""
    df = pd.DataFrame({"y": list(y_true), "p": list(p)})
    df["band"] = df["p"].apply(risk_band)
    out = df.groupby("band").agg(shipments=("y", "size"), actual_congestion_rate=("y", "mean"),
                                 avg_predicted=("p", "mean")).round(3)
    out["share_of_shipments"] = (out["shipments"] / len(df)).round(3)
    return out.reindex(["HIGH", "MEDIUM", "LOW"]).dropna(how="all")


if __name__ == "__main__":
    from train import ROOT, prepare, time_split, FEATURES, TARGET_CLF
    df = prepare(pd.read_csv(ROOT / "data" / "port_congestion_400.csv"))
    _, test = time_split(df)
    model = joblib.load(ROOT / "models" / "rf_clf_calibrated.joblib")
    p = model.predict_proba(test[FEATURES])[:, 1]
    summary = band_summary(test[TARGET_CLF], p)
    summary.to_csv(ROOT / "results" / "decision_bands.csv")
    print(summary.to_string())
