"""Test bed: a series of tests comparing the Regression approach vs the Random Forest approach.

Every test uses 5 dataset seeds (the data is regenerated each time) and reports mean +- std.
Outputs: results/*.csv and results/figures/*.png

Run:  python src/experiments.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data_generation import generate
from train import (prepare, time_split, build_model, build_calibrated, FEATURES,
                   TARGET_CLF, TARGET_REG, KIND)
from evaluate import clf_metrics, reg_metrics, plot_calibration

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEEDS = range(5)
CLF, REG = ["logreg", "rf_clf"], ["ridge", "rf_reg"]
THRESH = 12.0
LABEL = {"logreg": "Logistic Regression", "rf_clf": "Random Forest (clf)",
         "ridge": "Ridge Regression", "rf_reg": "Random Forest (reg)", "tree": "Decision Tree"}


def fit_eval(train, test, name, seed, depth=None):
    """Fit one model and return its metrics, or None if the test set has a single class."""
    if test[TARGET_CLF].nunique() < 2 or train[TARGET_CLF].nunique() < 2:
        return None
    m = build_model(name, seed, depth)
    if KIND[name] == "clf":
        m.fit(train[FEATURES], train[TARGET_CLF])
        return clf_metrics(test[TARGET_CLF], m.predict_proba(test[FEATURES])[:, 1])
    m.fit(train[FEATURES], train[TARGET_REG])
    pred = m.predict(test[FEATURES])
    out = reg_metrics(test[TARGET_REG], pred)
    out["auc"] = roc_auc_score(test[TARGET_CLF], pred)     # delay estimate used as a congestion score
    out["f1"] = clf_metrics(test[TARGET_CLF], 1 / (1 + np.exp(-(pred - THRESH))))["f1"]
    return out


def summarise(rows, by):
    df = pd.DataFrame(rows)
    return df.groupby(by).agg(["mean", "std"]).round(3)


def lines(df, x, y, path, title, xlabel, models):
    """Plot mean +- std of metric y against x for each model."""
    plt.figure(figsize=(6, 4))
    for m in models:
        d = df[df.model == m].groupby(x)[y].agg(["mean", "std"]).reset_index()
        plt.errorbar(d[x], d["mean"], yerr=d["std"], marker="o", capsize=3, label=LABEL[m])
    plt.xlabel(xlabel); plt.ylabel(y.upper()); plt.title(title)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def data(seed, **kw):
    return prepare(generate(seed=seed, **kw))


# T1 head-to-head
def t1_head_to_head():
    rows = []
    for s in SEEDS:
        tr, te = time_split(data(s))
        for m in CLF + REG + ["tree"]:
            r = fit_eval(tr, te, m, s, depth=4 if m == "tree" else None)
            if r: rows.append({"model": m, "seed": s, **r})
    df = pd.DataFrame(rows)
    out = df.drop(columns="seed").groupby("model").agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t1_head_to_head.csv"); print("\nT1 head-to-head\n", out.to_string())
    return df


# T2 data size
def t2_data_size():
    rows = []
    for n in [100, 200, 300, 400]:
        for s in SEEDS:
            tr, te = time_split(data(s, n=n))
            for m in CLF + REG:
                r = fit_eval(tr, te, m, s)
                if r: rows.append({"n_rows": n, "model": m, "seed": s, **r})
    df = pd.DataFrame(rows); df.to_csv(RES / "t2_data_size_raw.csv", index=False)
    lines(df, "n_rows", "auc", FIG / "t2_data_size.png", "T2: AUC vs dataset size", "Rows", CLF + ["ridge", "rf_reg"])
    out = df.drop(columns="seed").groupby(["n_rows", "model"])[["auc", "f1", "mae"]].agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t2_data_size.csv"); print("\nT2 data size\n", out.to_string())


# T3 noise / missing
def t3_noise():
    rows = []
    for lvl in [0.0, 0.1, 0.2, 0.3]:
        for s in SEEDS:
            # missing rate = lvl ; wind sensor noise sd = 40*lvl km/h (0 -> 12)
            tr, te = time_split(data(s, missing_rate=lvl, wind_noise_sd=40 * lvl))
            for m in CLF + REG:
                r = fit_eval(tr, te, m, s)
                if r: rows.append({"noise_level": lvl, "model": m, "seed": s, **r})
    df = pd.DataFrame(rows); df.to_csv(RES / "t3_noise_raw.csv", index=False)
    lines(df, "noise_level", "auc", FIG / "t3_noise.png", "T3: AUC vs missing/noise level",
          "Missing rate (wind noise scales with it)", CLF + ["ridge", "rf_reg"])
    out = df.drop(columns="seed").groupby(["noise_level", "model"])[["auc", "f1", "mae"]].agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t3_noise.csv"); print("\nT3 noise & missingness\n", out.to_string())


# T4 random vs time split
def t4_split_strategy():
    rows = []
    for s in SEEDS:
        df = data(s)
        splits = {"time": time_split(df),
                  "random": train_test_split(df, test_size=0.3, stratify=df[TARGET_CLF], random_state=s)}
        for sp, (tr, te) in splits.items():
            for m in CLF + REG:
                r = fit_eval(tr, te, m, s)
                if r: rows.append({"split": sp, "model": m, "seed": s, **r})
    df = pd.DataFrame(rows)
    out = df.drop(columns="seed").groupby(["split", "model"])[["auc", "f1", "mae"]].agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t4_split_strategy.csv"); print("\nT4 random vs time split\n", out.to_string())


# T5 leave-one-port-out
def t5_leave_port_out():
    rows = []
    for s in SEEDS:
        df = data(s)
        for port in df.port.unique():
            tr, te = df[df.port != port], df[df.port == port]
            for m in CLF + REG:
                r = fit_eval(tr, te, m, s)
                if r: rows.append({"held_out_port": port, "model": m, "seed": s, **r})
    df = pd.DataFrame(rows)
    out = df.drop(columns="seed").groupby(["held_out_port", "model"])[["auc", "f1", "mae"]].agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t5_leave_port_out.csv"); print("\nT5 leave-one-port-out\n", out.to_string())
    ov = df.drop(columns=["seed", "held_out_port"]).groupby("model")[["auc", "f1", "mae"]].mean().round(3)
    ov.to_csv(RES / "t5_overall.csv"); print("\nT5 overall\n", ov.to_string())



# T6 calibration
def t6_calibration():
    rows, curve = [], {}
    for s in SEEDS:
        tr, te = time_split(data(s))
        for m in CLF:
            raw = build_model(m, s).fit(tr[FEATURES], tr[TARGET_CLF])
            cal = build_calibrated(m, s).fit(tr[FEATURES], tr[TARGET_CLF])
            for tag, mod in [("uncalibrated", raw), ("calibrated", cal)]:
                p = mod.predict_proba(te[FEATURES])[:, 1]
                rows.append({"model": m, "version": tag, "seed": s, **clf_metrics(te[TARGET_CLF], p)})
                if s == 0: curve[f"{LABEL[m]} - {tag}"] = (te[TARGET_CLF], p)
    df = pd.DataFrame(rows)
    out = df.drop(columns="seed").groupby(["model", "version"])[["brier", "auc", "f1"]].agg(["mean", "std"]).round(3)
    out.to_csv(RES / "t6_calibration.csv"); print("\nT6 calibration\n", out.to_string())
    plot_calibration(curve, FIG / "t6_calibration.png", "T6: reliability curves (seed 0)")


# T7 tree depth sweep
def t7_depth_sweep():
    rows, rf = [], []
    for s in SEEDS:
        df = data(s); tr, te = time_split(df)
        for d in range(1, 16):
            m = build_model("tree", s, d).fit(tr[FEATURES], tr[TARGET_CLF])
            for tag, part in [("train", tr), ("test", te)]:
                rows.append({"depth": d, "set": tag, "seed": s,
                             "auc": roc_auc_score(part[TARGET_CLF], m.predict_proba(part[FEATURES])[:, 1])})
        f = build_model("rf_clf", s).fit(tr[FEATURES], tr[TARGET_CLF])
        rf.append(roc_auc_score(te[TARGET_CLF], f.predict_proba(te[FEATURES])[:, 1]))
    df = pd.DataFrame(rows)
    g = df.groupby(["depth", "set"])["auc"].agg(["mean", "std"]).round(3)
    g.to_csv(RES / "t7_depth_sweep.csv"); print("\nT7 depth sweep (mean AUC)\n", g["mean"].unstack().to_string())
    plt.figure(figsize=(6, 4))
    for tag in ["train", "test"]:
        d = df[df.set == tag].groupby("depth")["auc"].agg(["mean", "std"]).reset_index()
        plt.errorbar(d.depth, d["mean"], yerr=d["std"], marker="o", capsize=2, label=f"Single tree ({tag})")
    plt.axhline(np.mean(rf), color="g", ls="--", label=f"Random Forest test = {np.mean(rf):.3f}")
    plt.xlabel("Max depth"); plt.ylabel("AUC"); plt.title("T7: one tree vs a forest")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "t7_depth_sweep.png", dpi=140); plt.close()


if __name__ == "__main__":
    h = t1_head_to_head()
    t2_data_size(); t3_noise(); t4_split_strategy(); t5_leave_port_out(); t6_calibration(); t7_depth_sweep()
    # bar chart for T1
    plt.figure(figsize=(6, 4))
    d = h[h.model.isin(CLF + ["tree"])].groupby("model")["auc"].agg(["mean", "std"]).reindex(["logreg", "tree", "rf_clf"])
    plt.bar([LABEL[m] for m in d.index], d["mean"], yerr=d["std"], capsize=4, color=["#4C72B0", "#999", "#55A868"])
    plt.ylim(0.5, 1); plt.ylabel("Test AUC"); plt.title("T1: head-to-head (5 seeds)")
    plt.xticks(rotation=12); plt.tight_layout(); plt.savefig(FIG / "t1_head_to_head.png", dpi=140); plt.close()
    print("\nAll results saved to results/")
