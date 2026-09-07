"""只在本地审查包运行；从导出的开发输入重新拟合，不下载行情。"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from svxylab.features import CORE_IDS
from svxylab.prediction_data import TARGETS, development_csv
from svxylab.prediction_review import compare_runs
from svxylab.predictions import run_predictions


def main():
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256((ROOT / item["bundle_path"]).read_bytes()).hexdigest() == item["export_sha256"], item["bundle_path"]
    config = tomllib.loads((ROOT / "experiment.toml").read_text())
    definitions = json.loads((ROOT / "FEATURES.json").read_text())
    paths = manifest["development_inputs"]
    locked = config["data"]["locked_historical_start"]
    features = development_csv(ROOT / paths["features"], locked)
    availability = development_csv(ROOT / paths["availability"], locked)
    labels = development_csv(ROOT / paths["labels"], locked, labels=True)
    assert features.index.equals(labels.index) and features.index.equals(availability.index)
    assert features.columns.tolist() == CORE_IDS
    frame = features.join(labels).join(availability[["assumed_available_at"]])
    for key in CORE_IDS + TARGETS:
        frame[key] = pd.to_numeric(frame[key].replace("", np.nan), errors="raise")
    frame["observed"] = frame.observed.eq("True")
    for key in ["information_close_at", "decision_at", "execution_at", "label_matures_at", "label_available_at", "assumed_available_at"]:
        frame[key] = pd.to_datetime(frame[key], utc=True)
    assert frame.index.min() >= "2019-01-01" and frame.index.max() < locked
    assert frame.loc[frame.label_end_session.ge(locked), TARGETS].isna().all().all()
    assert (frame.decision_at == frame.assumed_available_at).all()
    output = ROOT / "replay_results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run_predictions(frame, config, definitions["interactions"], output)
    baseline = json.loads((ROOT / manifest["baseline_run"]).read_text())
    comparison = compare_runs(ROOT / baseline["result"]["output_dir"], output)
    # 绘图不参与数值复算；原图随包保存。
    (output / "refit_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
    audit = subprocess.run([sys.executable, "runs/p4/independent_validation.py", "--bundle", "."], cwd=ROOT,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (output / "saved_coefficient_audit.log").write_text(audit.stdout)
    assert audit.returncode == 0, audit.stdout
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--junitxml="+str(output / "pytest.xml")],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (output / "pytest.log").write_text(tests.stdout)
    assert tests.returncode == 0, tests.stdout
    evidence = {"method": "REFIT_FROM_EXPORTED_DEVELOPMENT_INPUTS", "forecast_sessions": result["forecast_sessions"],
                "refit_comparison_passed": True, "saved_coefficient_audit_exit_code": audit.returncode,
                "pytest_exit_code": tests.returncode, "raw_market_data_rebuilt": False,
                "manifest_files_verified": len(manifest["files"]), "output": output.relative_to(ROOT).as_posix()}
    (output / "offline_validation.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
