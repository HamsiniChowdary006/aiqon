"""Plain-English explanations of predictions.

  - Probability + confidence : calibrated Random Forest
  - Which factors drove it   : SHAP values on the (uncalibrated) Random Forest
  - Global cross-check       : Logistic Regression coefficients vs mean |SHAP|

Run:  python src/explain.py     (global importance + 3 example explanations)
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import ROOT, prepare, time_split, FEATURES, TARGET_CLF
from evaluate import confidence_label
from decision import recommend, risk_band

_cache = {}


def _models():
    if not _cache:
        _cache["cal"] = joblib.load(ROOT / "models" / "rf_clf_calibrated.joblib")
        _cache["rf"] = joblib.load(ROOT / "models" / "rf_clf.joblib")
        _cache["lr"] = joblib.load(ROOT / "models" / "logreg.joblib")
        _cache["explainer"] = shap.TreeExplainer(_cache["rf"].named_steps["model"])
    return _cache


def build_row(**raw):
    """Raw shipment fields -> model-ready one-row DataFrame (same features as training)."""
    raw = dict(raw)
    raw.setdefault("recent_avg_delay_hrs", np.nan)
    raw["arrival_time"] = pd.Timestamp("2025-01-01")      # only needed by prepare()
    return prepare(pd.DataFrame([raw]))


def _shap_values(X):
    m = _models()
    Xt = m["rf"].named_steps["prep"].transform(X[FEATURES])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    sv = m["explainer"].shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    names = [n.split("__", 1)[1] for n in m["rf"].named_steps["prep"].get_feature_names_out()]
    return sv, names


def _f(v, fmt="{:.0f}"):
    return "n/a" if pd.isna(v) else fmt.format(v)


def _vessels(v):
    """'1 vessel' / '5 vessels' / 'an unknown number of vessels'."""
    if pd.isna(v):
        return "an unknown number of vessels"
    return f"{int(v)} vessel" + ("" if int(v) == 1 else "s")


# feature -> text describing the ACTUAL state of the shipment
PHRASES = {
    "util": lambda r: f"queue is {_vessels(r.queue_length)} against {int(r.berth_capacity)} berths",
    "queue_length": lambda r: f"queue is {_vessels(r.queue_length)} against {int(r.berth_capacity)} berths",
    "berth_capacity": lambda r: f"the port has {int(r.berth_capacity)} berths",
    "incoming_ratio": lambda r: f"{_vessels(r.incoming_24h)} {'is' if int(r.incoming_24h) == 1 else 'are'} due within 24h for {int(r.berth_capacity)} berths",
    "incoming_24h": lambda r: f"{_vessels(r.incoming_24h)} {'is' if int(r.incoming_24h) == 1 else 'are'} due within 24h for {int(r.berth_capacity)} berths",
    "wind_speed_kmh": lambda r: (f"strong winds are expected ({_f(r.wind_speed_kmh)} km/h)"
                                 if r.wind_speed_kmh > 25 else f"winds are mild ({_f(r.wind_speed_kmh)} km/h)"),
    "rain_mm_hr": lambda r: (f"heavy rain is expected ({r.rain_mm_hr:.1f} mm/h)"
                             if r.rain_mm_hr > 4 else "rain is light or absent"),
    "is_night": lambda r: "arrival is at night (reduced pilot and labour availability)" if r.is_night else "arrival is in daytime",
    "hour": lambda r: "arrival is at night (reduced pilot and labour availability)" if r.is_night else "arrival is in daytime",
    "is_weekend": lambda r: "arrival is on a weekend" if r.is_weekend else "arrival is on a weekday",
    "day_of_week": lambda r: "arrival is on a weekend" if r.is_weekend else "arrival is on a weekday",
    "is_monsoon": lambda r: "it is monsoon season" if r.is_monsoon else "it is outside monsoon season",
    "month": lambda r: "it is monsoon season" if r.is_monsoon else "it is outside monsoon season",
    "vessel_size_teu": lambda r: f"the vessel is large ({r.vessel_size_teu:,.0f} TEU)" if r.vessel_size_teu > 8000
                                 else f"the vessel is {r.vessel_size_teu:,.0f} TEU",
    "recent_avg_delay_hrs": lambda r: f"the port's recent average delay is {_f(r.recent_avg_delay_hrs, '{:.1f}')} h",
}


def _phrase(feature, row):
    if feature.startswith("port_"):
        return f"{row.port} has its own typical delay level"
    return PHRASES[feature](row)


def _join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def explain(top_k=3, min_effect=0.01, **raw):
    """Return probability, confidence, band, actions and a plain-English sentence."""
    X = build_row(**raw)
    row = X.iloc[0]
    p = float(_models()["cal"].predict_proba(X[FEATURES])[0, 1])
    conf = str(confidence_label(p))
    band = risk_band(p)
    sv, names = _shap_values(X)
    contrib = pd.Series(sv[0], index=names).sort_values()

    def pick(series):
        out = []
        for feat, val in series.items():
            if abs(val) < min_effect:
                continue
            text = _phrase(feat, row)
            if text not in out:
                out.append(text)
            if len(out) == top_k:
                break
        return out

    up = pick(contrib[::-1][contrib[::-1] > 0])      # pushes risk up
    down = [d for d in pick(contrib[contrib < 0]) if d not in up]   # pushes risk down
    head = f"{band.capitalize()} risk ({p:.0%}, {conf.lower()} confidence): "
    if band == "LOW":
        sentence = head + (f"{_join(down)}." if down else "no major risk drivers.")
    else:
        sentence = head + (f"{_join(up)}." if up else "no single dominant driver.")
        if down:
            sentence += f" Mitigating: {_join(down)}."
    return {"probability": p, "confidence": conf, "band": band,
            "drivers_up": up, "drivers_down": down, "sentence": sentence,
            "actions": recommend(p, conf)["actions"], "shap": contrib}


def global_importance():
    """Mean |SHAP| of the RF vs |coefficient| of Logistic Regression, grouped by raw feature."""
    m = _models()
    df = prepare(pd.read_csv(ROOT / "data" / "port_congestion_400.csv"))
    _, test = time_split(df)
    sv, names = _shap_values(test)
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=names)
    lr = m["lr"]
    lr_names = [n.split("__", 1)[1] for n in lr.named_steps["prep"].get_feature_names_out()]
    lr_imp = pd.Series(np.abs(lr.named_steps["model"].coef_[0]), index=lr_names)

    def group(s):   # merge port one-hot columns into one "port" feature
        s = s.copy(); s.index = ["port" if i.startswith("port_") else i for i in s.index]
        return s.groupby(level=0).sum()

    tab = pd.DataFrame({"rf_mean_abs_shap": group(shap_imp), "logreg_abs_coef": group(lr_imp)})
    tab["rf_share"] = (tab.rf_mean_abs_shap / tab.rf_mean_abs_shap.sum()).round(3)
    tab["logreg_share"] = (tab.logreg_abs_coef / tab.logreg_abs_coef.sum()).round(3)
    tab["rf_rank"] = tab.rf_mean_abs_shap.rank(ascending=False).astype(int)
    tab["logreg_rank"] = tab.logreg_abs_coef.rank(ascending=False).astype(int)
    tab = tab.sort_values("rf_rank").round(4)
    tab.to_csv(ROOT / "results" / "global_importance.csv")

    top = tab.head(10)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].barh(top.index, top.rf_share, color="#55A868"); ax[0].set_title("Random Forest: share of mean |SHAP|")
    ax[1].barh(top.index, top.logreg_share, color="#4C72B0"); ax[1].set_title("Logistic Regression: share of |coef|")
    plt.tight_layout(); plt.savefig(ROOT / "results" / "figures" / "global_importance.png", dpi=140); plt.close()
    overlap = len(set(tab.nsmallest(5, "rf_rank").index) & set(tab.nsmallest(5, "logreg_rank").index))
    return tab, overlap


if __name__ == "__main__":
    tab, overlap = global_importance()
    print("Top 8 features (rank by RF SHAP):\n", tab.head(8)[["rf_share", "logreg_share", "rf_rank", "logreg_rank"]].to_string())
    print(f"\nTop-5 overlap between RF and Logistic: {overlap}/5")

    df = prepare(pd.read_csv(ROOT / "data" / "port_congestion_400.csv"))
    _, test = time_split(df)
    p = _models()["cal"].predict_proba(test[FEATURES])[:, 1]
    picks = {"HIGH example": int(np.argmax(p)), "MEDIUM example": int(np.argmin(np.abs(p - 0.55))),
             "LOW example": int(np.argmin(p))}
    cols = ["port", "hour", "day_of_week", "month", "berth_capacity", "queue_length", "incoming_24h",
            "wind_speed_kmh", "rain_mm_hr", "vessel_size_teu", "recent_avg_delay_hrs"]
    lines = []
    for label, i in picks.items():
        r = test.iloc[i]
        out = explain(**{c: r[c] for c in cols})
        lines += [f"[{label}] actual delay {r.delay_hours:.1f} h (congested={int(r.congested)})",
                  out["sentence"], "Actions: " + "; ".join(out["actions"]), ""]
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "results" / "example_explanations.txt").write_text(text, encoding="utf-8")