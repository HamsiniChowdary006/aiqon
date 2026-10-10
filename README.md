# Port Congestion Risk Prediction

Predicts the likelihood that a vessel will wait more than 12 hours before berthing, gives a confidence level, explains the key factors in plain English, and maps the risk to an operational action.

Built for the Aiqon AI Internship take-home assignment. **Full method, results and discussion are in the technical report: [`report/report.pdf`](report/report.pdf).**

> **Note:** all results come from a **synthetic dataset** with known rules. They show how the methods behave on that simulation, not performance on real port or AIS data.

## Overview

| Step | What it does |
|---|---|
| **Define congestion** | Waiting time before berthing above 12 hours |
| **Data** | 400 simulated vessel arrivals, 4 ports, 12 months, with noise and missing values |
| **Models** | Regression approach (Logistic, Ridge) vs Random Forest (classifier, regressor) |
| **Test bed** | 7 tests over 5 seeds: head-to-head, data size, noise, split strategy, unseen port, calibration, tree depth |
| **Output** | Probability, Low/Medium/High confidence, SHAP-based explanation, recommended action |

Example output:

```
Medium risk (53%, low confidence): strong winds are expected (28 km/h), the port's recent
average delay is 12.8 h and arrival is on a weekend. Mitigating: it is outside monsoon
season, queue is 1 vessel against 4 berths and 3 vessels are due within 24h for 4 berths.
```

Decision bands: **above 70%** reroute or slow-steam, alert the consignee, rebook trucking; **40 to 70%** monitor and keep buffer time; **below 40%** proceed normally.

## Quick start

Run from the project root (tested on Python 3.13, Windows).

```bash
pip install -r requirements.txt

python src/data_generation.py   # data/port_congestion_400.csv
python src/train.py             # models/ + results/final_metrics.csv
python src/experiments.py       # test bed -> results/
python src/decision.py          # risk-band table
python src/explain.py           # SHAP importance + example explanations
python src/eda.py               # exploratory figures

streamlit run app.py            # demo at http://localhost:8501
```

- For the demo you only need the first three steps.
- If `streamlit` is not recognised on Windows, use `python -m streamlit run app.py`.
- `models/` is not committed, so run `python src/train.py` after cloning.
- Seeds are fixed, so reruns reproduce the same numbers.

## Repository structure

```
port-congestion/
├── app.py                 # Streamlit demo
├── requirements.txt
├── data/                  # synthetic dataset (generated)
├── src/
│   ├── data_generation.py # simulation, ground truth, noise
│   ├── train.py           # features, pipelines, time-based split
│   ├── evaluate.py        # metrics, confidence labels, plots
│   ├── experiments.py     # test bed (T1 to T7)
│   ├── explain.py         # SHAP + plain-English explanations
│   ├── decision.py        # risk band -> action
│   └── eda.py             # exploratory plots
├── results/               # CSV tables and figures
├── models/                # saved models (not in git)
└── report/                # technical report (PDF)
```

## Headline results

Time-based split (train on earlier arrivals, test on later), mean over 5 seeds, synthetic data.

| Model | Test AUC | Delay MAE (h) |
|---|---|---|
| Logistic / Ridge Regression | 0.865 | 2.22 |
| Random Forest | 0.864 | 2.07 |
| Single Decision Tree | 0.755 | n/a |

The two approaches are statistically tied for classification, and a single tree is clearly worse. The report covers the other six tests, calibration, explainability and the decision bands in detail.

![EDA overview](results/figures/eda_overview.png)

## Key limitations

- The models learn my simulation rules, so scores do not prove real-world performance.
- With 400 rows, most gaps between models fall within one standard deviation.
- Calibration did not improve Brier scores in this setup.
- Explanations show what the model relies on, not causal effects.

See the report for assumptions and what would change with real AIS data.

## Troubleshooting

| Problem | Fix |
|---|---|
| App says "Trained models not found" | Run `python src/train.py`, then reload |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `streamlit` not recognised | Use `python -m streamlit run app.py` |