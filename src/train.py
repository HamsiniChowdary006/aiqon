"""Features, model pipelines, time-based split, and final training run.

Two approaches are compared, each as a classifier and a delay regressor:
  Regression approach : LogisticRegression (P(congested))  |  Ridge (delay hours)
  Random Forest       : RandomForestClassifier             |  RandomForestRegressor
A single DecisionTree is kept only for the depth-sweep test (one tree vs a forest).

Hyper-parameters are fixed, sensible defaults (never tuned on the test set).

Run:  python src/train.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from evaluate import clf_metrics, reg_metrics, plot_confusion, plot_calibration
from data_generation import generate

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["hour", "day_of_week", "berth_capacity", "queue_length", "incoming_24h",
           "wind_speed_kmh", "rain_mm_hr", "vessel_size_teu", "recent_avg_delay_hrs",
           "util", "incoming_ratio", "is_night", "is_weekend", "is_monsoon"]
# "month" is deliberately NOT a feature: in a time-based split the test months are unseen,
# so the model would use it as a time proxy. is_monsoon carries the seasonal signal instead.
CATEGORICAL = ["port"]
FEATURES = NUMERIC + CATEGORICAL
TARGET_CLF, TARGET_REG = "congested", "delay_hours"
KIND = {"logreg": "clf", "tree": "clf", "rf_clf": "clf", "ridge": "reg", "rf_reg": "reg"}


def prepare(df):
    """Sort by time and add engineered features (all computed from the row itself only)."""
    df = df.copy()
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df = df.sort_values("arrival_time").reset_index(drop=True)
    df["util"] = df["queue_length"] / df["berth_capacity"]          # queue pressure
    df["incoming_ratio"] = df["incoming_24h"] / df["berth_capacity"]  # arrival density
    df["is_night"] = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    return df


def time_split(df, frac=0.7):
    """Earlier rows -> train, later rows -> test (no shuffling)."""
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def _preprocessor(scale):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:                       # linear models need scaling; trees do not
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def build_model(name, seed=0, depth=None):
    """Return an sklearn Pipeline (preprocessing is fit on training data only)."""
    if name == "logreg":
        est, scale = LogisticRegression(max_iter=2000), True
    elif name == "ridge":
        est, scale = Ridge(alpha=1.0), True
    elif name == "tree":
        est, scale = DecisionTreeClassifier(max_depth=depth, random_state=seed), False
    elif name == "rf_clf":
        est, scale = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                            random_state=seed, n_jobs=-1), False
    elif name == "rf_reg":
        est, scale = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                           random_state=seed, n_jobs=-1), False
    else:
        raise ValueError(name)
    return Pipeline([("prep", _preprocessor(scale)), ("model", est)])


def build_calibrated(name, seed=0):
    """Sigmoid calibration using time-ordered folds (isotonic overfits on ~300 rows)."""
    return CalibratedClassifierCV(build_model(name, seed), method="sigmoid",
                                  cv=TimeSeriesSplit(n_splits=3))


def main():
    for folder in ["data", "models", "results", "results/figures"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    csv = ROOT / "data" / "port_congestion_400.csv"
    if not csv.exists():                       # rebuild dataset if the CSV is missing
        generate().to_csv(csv, index=False)
        print("Dataset not found, generated", csv)
    df = prepare(pd.read_csv(csv))
    train, test = time_split(df)
    print(f"Train {len(train)} rows ({train.arrival_time.min().date()} to {train.arrival_time.max().date()}) | "
          f"Test {len(test)} rows ({test.arrival_time.min().date()} to {test.arrival_time.max().date()})")
    print(f"Congestion rate  train {train[TARGET_CLF].mean():.2f} | test {test[TARGET_CLF].mean():.2f}")

    rows, probs = [], {}
    for name, model in [("logreg", build_model("logreg")), ("rf_clf", build_model("rf_clf", 42)),
                        ("logreg_calibrated", build_calibrated("logreg")),
                        ("rf_clf_calibrated", build_calibrated("rf_clf", 42))]:
        model.fit(train[FEATURES], train[TARGET_CLF])
        p = model.predict_proba(test[FEATURES])[:, 1]
        probs[name] = p
        rows.append({"model": name, **clf_metrics(test[TARGET_CLF], p)})
    for name in ["ridge", "rf_reg"]:
        m = build_model(name, 42).fit(train[FEATURES], train[TARGET_REG])
        rows.append({"model": name, **reg_metrics(test[TARGET_REG], m.predict(test[FEATURES]))})
        joblib.dump(m, ROOT / "models" / f"{name}.joblib")

    res = pd.DataFrame(rows).set_index("model").round(3)
    print(res.to_string())
    res.to_csv(ROOT / "results" / "final_metrics.csv")

    fig = ROOT / "results" / "figures"
    plot_confusion(test[TARGET_CLF], probs["rf_clf_calibrated"], fig / "confusion_rf_calibrated.png",
                   "RF (calibrated), test set")
    plot_calibration({k: (test[TARGET_CLF], v) for k, v in probs.items()}, fig / "final_calibration.png")

    # models saved for explain.py and the Streamlit demo (trained on the training period only)
    for name, mk in [("logreg_calibrated", lambda: build_calibrated("logreg")),
                     ("rf_clf_calibrated", lambda: build_calibrated("rf_clf", 42))]:
        joblib.dump(mk().fit(train[FEATURES], train[TARGET_CLF]), ROOT / "models" / f"{name}.joblib")
    # uncalibrated RF is kept separately because SHAP needs the raw forest
    joblib.dump(build_model("rf_clf", 42).fit(train[FEATURES], train[TARGET_CLF]),
                ROOT / "models" / "rf_clf.joblib")
    joblib.dump(build_model("logreg").fit(train[FEATURES], train[TARGET_CLF]),
                ROOT / "models" / "logreg.joblib")
    print("Saved models to models/")


if __name__ == "__main__":
    main()