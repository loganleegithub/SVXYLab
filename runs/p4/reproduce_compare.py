"""P4 首月、首次完整运行及最终完整运行的精确复算比较。"""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FIELDS = {"computed_at_utc", "elapsed_seconds"}


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in RUNTIME_FIELDS}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def csv_rows(path):
    with path.open(newline="") as source:
        return [normalized(row) for row in csv.DictReader(source)]


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    records = [(p, json.loads(p.read_text())) for p in sorted((ROOT / "runs/p4").glob("*/predictions_run.json"))]
    pilot_path, pilot = next((p, d) for p, d in records if d["result"]["pilot"])
    full = [(p, d) for p, d in records if not d["result"]["pilot"]]
    assert len(full) >= 2
    initial_path, initial = full[0]
    final_path, final = full[-1]
    old_dir = ROOT / initial["result"]["output_dir"]
    new_dir = ROOT / final["result"]["output_dir"]
    comparisons = []
    for old in sorted(old_dir.glob("*.csv")):
        new = new_dir / old.name
        a, b = csv_rows(old), csv_rows(new)
        assert a == b, old.name
        comparisons.append({"file": old.name, "rows": len(a), "equal": True,
                            "normalized_sha256": digest(a),
                            "byte_equal": old.read_bytes() == new.read_bytes()})
    snapshots = {}
    for kind in ["models", "selections"]:
        sources = sorted((old_dir / kind).glob("*.json"))
        assert len(sources) == len(list((new_dir / kind).glob("*.json")))
        for old in sources:
            a = normalized(json.loads(old.read_text()))
            b = normalized(json.loads((new_dir / kind / old.name).read_text()))
            assert a == b, str(old)
        snapshots[kind] = {"count": len(sources), "parameters_and_transformations_exactly_equal": True}
    p = csv_rows(ROOT / pilot["result"]["output_dir"] / "predictions.csv")
    a = [r for r in csv_rows(old_dir / "predictions.csv")
         if r["decision_session"] <= pilot["result"]["processed_decision_last"]]
    assert p == a
    assert initial["input_verification"] == final["input_verification"]
    algorithm = ["prediction_data.py", "models.py", "predictions.py"]
    for name in algorithm:
        key = f"src/svxylab/{name}"
        assert initial["source_sha256"][key] == final["source_sha256"][key]
    unchanged_spec = {}
    for name in ["experiment.toml", "FEATURES.json"]:
        original = subprocess.check_output(["git", "show", f"541178b:{name}"], cwd=ROOT)
        assert original == (ROOT / name).read_bytes()
        unchanged_spec[name] = sha256(original).hexdigest()
    future_path = initial_path.parent / "real_future_perturbation.json"
    future = json.loads(future_path.read_text())
    assert future["passed"] and future["frozen_input_hashes_unchanged"]
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "command": ".venv/bin/python runs/p4/reproduce_compare.py",
        "comparison_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "pilot_run": pilot_path.relative_to(ROOT).as_posix(),
        "initial_run": initial_path.relative_to(ROOT).as_posix(),
        "final_run": final_path.relative_to(ROOT).as_posix(),
        "runtime_fields_excluded": sorted(RUNTIME_FIELDS),
        "csv_comparisons": comparisons,
        "snapshot_comparisons": snapshots,
        "pilot_first_month_prediction_rows_exactly_equal": len(p),
        "forecast_model_rows": 3 * final["result"]["forecast_sessions"],
        "frozen_inputs_identical": True,
        "algorithm_sources_identical": algorithm,
        "unchanged_accepted_specification": unchanged_spec,
        "future_perturbation_record": future_path.relative_to(ROOT).as_posix(),
        "future_test_applies_to_initial_run_with_identical_final_predictions": True,
        "passed": True,
    }
    (ROOT / "runs/p4/reproducibility.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "csv_files": len(comparisons), "snapshots": snapshots,
                      "pilot_rows": len(p), "final_run": result["final_run"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
