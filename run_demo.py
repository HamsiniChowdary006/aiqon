"""One-command launcher.

    python run_demo.py            # install missing packages, build data + models if needed, open the demo
    python run_demo.py --all      # also reproduce every result (test bed, risk bands, SHAP, EDA), then open the demo
    python run_demo.py --retrain  # rebuild the models even if they already exist
    python run_demo.py --no-app   # do the setup/analysis steps but do not start the demo

Works on Windows, macOS and Linux (only needs Python).
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
            "joblib": "joblib", "shap": "shap", "streamlit": "streamlit"}


def run(args, label):
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also run experiments, decision, explain and eda")
    ap.add_argument("--retrain", action="store_true", help="retrain models even if models/ exists")
    ap.add_argument("--no-app", action="store_true", help="do not launch the Streamlit demo")
    a = ap.parse_args()

    # 1. packages
    missing = [pkg for mod, pkg in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        run(["-m", "pip", "install", "-r", "requirements.txt"], f"Installing missing packages: {', '.join(missing)}")
    else:
        print("=== Packages: all installed ===")

    # 2. dataset
    csv = ROOT / "data" / "port_congestion_400.csv"
    if not csv.exists():
        (ROOT / "data").mkdir(exist_ok=True)
        run(["src/data_generation.py"], "Generating the 400-row dataset")
    else:
        print("=== Dataset: found ===")

    # 3. models
    if a.retrain or not (ROOT / "models" / "rf_clf_calibrated.joblib").exists():
        run(["src/train.py"], "Training models")
    else:
        print("=== Models: found (use --retrain to rebuild) ===")

    # optional: reproduce all results
    if a.all:
        run(["src/experiments.py"], "Running the test bed (takes a minute or two)")
        run(["src/decision.py"], "Building risk-band table")
        run(["src/explain.py"], "Computing SHAP explanations")
        run(["src/eda.py"], "Creating EDA figures")

    # 4. demo
    if a.no_app:
        print("\nDone. Start the demo with:  python -m streamlit run app.py")
        return
    print("\n=== Starting the demo at http://localhost:8501  (press Ctrl+C to stop) ===")
    print("If Streamlit asks for an email, just press Enter.\n", flush=True)
    subprocess.run([PY, "-m", "streamlit", "run", "app.py"], cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nA step failed (exit code {e.returncode}). See the error above.")
    except KeyboardInterrupt:
        print("\nStopped.")
