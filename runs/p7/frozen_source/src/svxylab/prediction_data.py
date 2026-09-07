"""P4 冻结输入与时间边界；任何拟合都通过同一成熟条件。"""

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from svxylab.features import CORE_IDS
from svxylab.features_report import verify_hashes

TARGETS = ["R5", "L5", "Y10"]


def development_csv(path: Path, locked_start: str, *, labels=False) -> pd.DataFrame:
    """截至封存边界读行；跨界标签不转成数值，也不用于评分。"""
    selected = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["as_of_session"] >= locked_start:
                break
            if labels and row["label_end_session"] >= locked_start:
                for key in TARGETS:
                    row[key] = ""
                row["observed"] = "False"
                row["reason"] = "BEYOND_DEVELOPMENT_OUTCOME_BOUNDARY"
            selected.append(row)
    return pd.DataFrame(selected).set_index("as_of_session")


def load_development_data(root: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    p3_accept = json.loads((root / "runs/p3/acceptance.json").read_text())
    verify_hashes(root, {p3_accept["accepted_run"]: p3_accept["accepted_run_sha256"]})
    verify_hashes(root, p3_accept["accepted_artifact_sha256"])
    p3 = json.loads((root / p3_accept["accepted_run"]).read_text())
    p2_accept = json.loads((root / "runs/p2/acceptance.json").read_text())
    verify_hashes(root, {p2_accept["accepted_run"]: p2_accept["accepted_run_sha256"]})
    verify_hashes(root, p2_accept["accepted_artifact_sha256"])
    p2 = json.loads((root / p2_accept["accepted_run"]).read_text())
    p1 = json.loads((root / "runs/p1/reproducibility.json").read_text())
    verify_hashes(root, p1["clean_sha256"])
    manifest = root / "data/raw/downloads.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    verify_hashes(root, {r["raw_file"]: r["sha256"] for r in records})
    locked = config["data"]["locked_historical_start"]
    p3_dir, p2_dir = root / p3["result"]["output_dir"], root / p2["result"]["output_dir"]
    features = development_csv(p3_dir / "core_features.csv", locked)
    available = development_csv(p3_dir / "feature_availability.csv", locked)
    labels = development_csv(p2_dir / "labels.csv", locked, labels=True)
    if not features.index.equals(labels.index) or not features.index.equals(available.index):
        raise ValueError("P2 标签与 P3 特征必须保留同一完整交易日日历")
    if features.columns.tolist() != CORE_IDS or features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("P4 输入必须是已验收的19列核心和唯一递增交易日")
    frame = features.join(labels).join(available[["assumed_available_at"]])
    frame = frame.loc[frame.index >= config["data"]["feature_history_requested_start"]].copy()
    for key in CORE_IDS + TARGETS:
        frame[key] = pd.to_numeric(frame[key].replace("", np.nan), errors="raise")
    frame["observed"] = frame.observed.eq("True")
    for key in ["information_close_at", "decision_at", "execution_at", "label_matures_at", "label_available_at", "assumed_available_at"]:
        frame[key] = pd.to_datetime(frame[key], utc=True)
    if not (frame.decision_at == frame.assumed_available_at).all():
        raise ValueError("P3 特征可用时点与 P2 决策时钟不一致")
    finite = frame[TARGETS].notna().all(axis=1)
    if not (frame.loc[finite, "L5"].between(0, 1).all() and frame.loc[finite, "Y10"].isin([0, 1]).all()
            and frame.loc[finite, "R5"].ge(-1).all()):
        raise ValueError("已验收标签的值域无效")
    return frame, {"p3_accepted_run": p3_accept["accepted_run"], "p2_accepted_run": p2_accept["accepted_run"],
                   "p3_artifact_sha256": p3_accept["accepted_artifact_sha256"],
                   "p2_artifact_sha256": p2_accept["accepted_artifact_sha256"], "p1_clean_sha256": p1["clean_sha256"],
                   "raw_records_verified": len(records), "raw_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                   "development_information_first": frame.index[0], "development_information_last": frame.index[-1],
                   "development_information_rows": len(frame), "locked_start": locked,
                   "label_values_from_locked_period_used": False}


def training_rows(frame: pd.DataFrame, position: int, config: dict, *, cutoff=None) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff if cutoff is not None else frame.decision_at.iloc[position])
    if cutoff.tzinfo is None:
        raise ValueError("训练截止必须带时区")
    window = frame.iloc[max(0, position - config["training"]["max_window_sessions"] + 1):position + 1]
    complete = np.isfinite(window[CORE_IDS + TARGETS].to_numpy(dtype=float)).all(axis=1)
    allowed = (complete & window.observed & window.assumed_available_at.le(cutoff)
               & window.label_matures_at.lt(cutoff) & window.label_available_at.le(cutoff))
    return window.loc[allowed].copy()


def inner_splits(frame: pd.DataFrame, outer: pd.DataFrame, config: dict) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    if outer.empty:
        return []
    count = config["training"]["inner_validation_blocks"]
    width = config["training"]["inner_validation_block_sessions"]
    last = frame.index.get_loc(outer.index[-1])
    first = last + 1 - count * width
    if first < 0:
        return []
    splits = []
    for block in range(count):
        dates = frame.index[first + block * width:first + (block + 1) * width]
        cutoff = frame.loc[dates[0], "decision_at"]
        train = outer.loc[(outer.index < dates[0]) & outer.label_matures_at.lt(cutoff)
                          & outer.label_available_at.le(cutoff) & outer.assumed_available_at.le(cutoff)].copy()
        validation = outer.loc[outer.index.isin(dates)].copy()
        info = {"block": block + 1, "validation_information_first": dates[0], "validation_information_last": dates[-1],
                "validation_cutoff": cutoff.isoformat(), "train_rows": len(train), "validation_rows": len(validation),
                "train_positive_count": int(train.Y10.sum()), "train_event_clusters": event_clusters(train),
                "validation_positive_count": int(validation.Y10.sum()),
                "train_first": train.index[0] if len(train) else None, "train_last": train.index[-1] if len(train) else None,
                "train_latest_label_matures_at": train.label_matures_at.max().isoformat() if len(train) else None,
                "train_latest_label_available_at": train.label_available_at.max().isoformat() if len(train) else None}
        splits.append((train, validation, info))
    return splits


def eligibility(frame: pd.DataFrame, position: int, config: dict, *, need_selection: bool) -> tuple[pd.DataFrame, list, str]:
    train = training_rows(frame, position, config)
    if not np.isfinite(frame[CORE_IDS].iloc[position].to_numpy(dtype=float)).all():
        return train, [], "MISSING_CORE_FEATURES"
    if train.empty or train.index[-1] < config["data"]["initial_train_end"]:
        return train, [], "BEFORE_INITIAL_TRAIN_END"
    if len(train) < config["training"]["min_outer_training_rows"]:
        return train, [], "INSUFFICIENT_OUTER_TRAINING_ROWS"
    splits = inner_splits(frame, train, config) if need_selection else []
    if need_selection and (len(splits) != config["training"]["inner_validation_blocks"] or
                           any(len(t) < config["training"]["min_inner_training_rows"] or v.empty for t,v,_ in splits)):
        return train, splits, "INSUFFICIENT_INNER_TRAINING_ROWS"
    return train, splits, "ELIGIBLE"


def event_clusters(train: pd.DataFrame) -> int:
    """合并阳性标签的重叠入场至终点区间，仅为样本依赖诊断。"""
    end = ""
    clusters = 0
    for row in train.loc[train.Y10.eq(1)].itertuples():
        if row.execution_session > end:
            clusters += 1
        end = max(end, row.label_end_session)
    return clusters
