"""M0/M1/M2：仅在调用者提供的成熟训练样本拟合，状态可保存为普通 JSON。"""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import log_loss, mean_pinball_loss, mean_squared_error
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from svxylab.features import CORE_IDS

MODELS = ["M0", "M1", "M2"]
HEADS = ["mu5", "q90", "p10"]
LABEL_FOR_HEAD = {"mu5": "R5", "q90": "L5", "p10": "Y10"}
GRID_FOR_HEAD = {"mu5": "ridge_alpha_grid", "q90": "quantile_alpha_grid", "p10": "logistic_C_grid"}


def scaler_snapshot(scaler):
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "variance": scaler.var_.tolist()}


class Design:
    def __init__(self, model, config, interactions):
        self.model, self.config, self.interactions = model, config, interactions
        self.raw_scaler = StandardScaler()
        self.output_scaler = None
        self.spline = None
        self.pairs = [(CORE_IDS.index(x["features"][0]), CORE_IDS.index(x["features"][1])) for x in interactions]

    def fit(self, raw):
        if not np.isfinite(raw).all():
            raise ValueError("训练特征有缺失或非有限值")
        self.raw_min, self.raw_max = np.min(raw, axis=0), np.max(raw, axis=0)
        z = self.raw_scaler.fit_transform(raw)
        self.z_quantiles = np.quantile(z, [.25, .5, .75], axis=0, method="linear")
        if self.model == "M1":
            self.columns = CORE_IDS.copy()
            return z
        spec = self.config["models"]
        self.spline = SplineTransformer(n_knots=spec["spline_knots"], degree=spec["spline_degree"],
                                        knots=spec["spline_knot_placement"], include_bias=spec["spline_include_bias"],
                                        extrapolation=spec["spline_extrapolation"])
        splines = self.spline.fit_transform(z)
        self.basis_per_feature = splines.shape[1] // len(CORE_IDS)
        self.columns = [f"{key}_spline_{j}" for key in CORE_IDS for j in range(self.basis_per_feature)]
        self.columns += [x["id"] for x in self.interactions]
        products = np.column_stack([z[:, a] * z[:, b] for a,b in self.pairs])
        self.output_scaler = StandardScaler()
        return self.output_scaler.fit_transform(np.column_stack([splines, products]))

    def transform(self, raw):
        z = self.raw_scaler.transform(raw)
        if self.model == "M1":
            return z
        products = np.column_stack([z[:, a] * z[:, b] for a,b in self.pairs])
        return self.output_scaler.transform(np.column_stack([self.spline.transform(z), products]))

    def snapshot(self):
        result = {"model": self.model, "input_columns": CORE_IDS, "columns": self.columns,
                  "raw_scaler": scaler_snapshot(self.raw_scaler), "raw_min": self.raw_min.tolist(), "raw_max": self.raw_max.tolist(),
                  "training_z_quantiles_25_50_75": self.z_quantiles.tolist(), "interactions": self.interactions}
        if self.model == "M2":
            result.update({"output_scaler": scaler_snapshot(self.output_scaler), "basis_per_feature": self.basis_per_feature,
                           "extrapolation": "linear", "include_bias": False,
                           "bsplines": [{"knots": s.t.tolist(), "coefficients": s.c.tolist(), "degree": s.k} for s in self.spline.bsplines_]})
        return result

    def support(self, raw):
        outside = (raw < self.raw_min) | (raw > self.raw_max)
        distance = np.maximum(np.maximum(self.raw_min-raw, raw-self.raw_max), 0) / self.raw_scaler.scale_
        return outside, np.max(distance, axis=1)


def constant_head(head, y, quantile):
    if head == "mu5":
        value = np.mean(y)
    elif head == "q90":
        value = np.quantile(y, quantile, method="linear")
    else:
        value = (np.sum(y)+.5)/(len(y)+1)
    return {"kind": "constant", "value": float(value), "rows": len(y), "positive_count": int(np.sum(y)) if head == "p10" else None}


def fit_head(head, X, y, parameter, config):
    start = perf_counter()
    quantile = config["targets"]["loss_quantile"]
    if head == "p10" and len(np.unique(y)) < 2:
        return {**constant_head(head, y, quantile), "fallback": "SINGLE_CLASS_JEFFREYS", "parameter": parameter,
                "elapsed_seconds": perf_counter()-start, "warnings": [], "iterations": 0}
    if head == "mu5":
        estimator = Ridge(alpha=parameter, solver="svd", fit_intercept=True)
    elif head == "p10":
        estimator = LogisticRegression(C=parameter, l1_ratio=0.0, solver="lbfgs", class_weight=None,
                                       tol=1e-8, max_iter=3000, random_state=config["training"]["seed"])
    else:
        estimator = QuantileRegressor(quantile=quantile, alpha=parameter, solver="highs", fit_intercept=True)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        estimator.fit(X, y)
    messages = [str(w.message) for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError(f"{head} 拟合未收敛：{messages}")
    coef = np.asarray(estimator.coef_).reshape(-1)
    intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    if not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError(f"{head} 拟合得到非有限系数")
    iterations = getattr(estimator, "n_iter_", None)
    return {"kind": "logistic" if head == "p10" else "linear", "coef": coef.tolist(), "intercept": intercept,
            "parameter": float(parameter), "rows": len(y), "positive_count": int(np.sum(y)) if head == "p10" else None,
            "iterations": int(np.max(iterations)) if iterations is not None else None,
            "warnings": messages, "elapsed_seconds": perf_counter()-start, "fallback": None}


def head_score(head, fitted, design, n):
    if fitted["kind"] == "constant":
        value = logit(fitted["value"]) if head == "p10" else fitted["value"]
        return np.full(n, value, dtype=float)
    return design @ np.asarray(fitted["coef"]) + fitted["intercept"]


def publish(scores):
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError("模型产生非有限预测，不能填为零")
    return {"mu5_raw": scores["mu5"], "mu5": np.maximum(scores["mu5"], -1),
            "q90_raw": scores["q90"], "q90": np.clip(scores["q90"], 0, 1),
            "p10_logit": scores["p10"], "p10": expit(scores["p10"]),
            "mu_projected": scores["mu5"] < -1, "q_projected": (scores["q90"] < 0) | (scores["q90"] > 1)}


def prediction_loss(head, y, raw_score, quantile=.9):
    if head == "mu5":
        return float(mean_squared_error(y, np.maximum(raw_score, -1)))
    if head == "q90":
        return float(mean_pinball_loss(y, np.clip(raw_score, 0, 1), alpha=quantile))
    return float(log_loss(y, expit(raw_score), labels=[0, 1]))


@dataclass
class JointModel:
    model: str
    design: Design | None
    heads: dict

    def scores(self, raw):
        design = self.design.transform(raw) if self.design is not None else None
        return {h: head_score(h, state, design, len(raw)) for h,state in self.heads.items()}

    def predict(self, raw):
        return publish(self.scores(raw))

    def snapshot(self):
        return {"model": self.model, "transform": self.design.snapshot() if self.design else None, "heads": self.heads}


def fit_joint(model, train, parameters, config, interactions):
    raw = train[CORE_IDS].to_numpy(dtype=float)
    design = Design(model, config, interactions) if model != "M0" else None
    X = design.fit(raw) if design else None
    heads = {}
    for head in HEADS:
        y = train[LABEL_FOR_HEAD[head]].to_numpy(dtype=float)
        heads[head] = (fit_head(head, X, y, parameters[head], config) if model != "M0"
                       else {**constant_head(head, y, config["targets"]["loss_quantile"]), "fallback": None,
                             "parameter": None, "warnings": [], "iterations": 0, "elapsed_seconds": 0.0})
    return JointModel(model, design, heads)


def select_parameters(splits, model, config, interactions):
    scores = []
    transforms = []
    for train, validation, info in splits:
        design = Design(model, config, interactions)
        X = design.fit(train[CORE_IDS].to_numpy(dtype=float))
        V = design.transform(validation[CORE_IDS].to_numpy(dtype=float))
        transforms.append({**info, "transform": design.snapshot()})
        for head in HEADS:
            grid = sorted(config["models"][GRID_FOR_HEAD[head]], reverse=head != "p10")
            for parameter in grid:
                fitted = fit_head(head, X, train[LABEL_FOR_HEAD[head]].to_numpy(dtype=float), parameter, config)
                raw_score = head_score(head, fitted, V, len(V))
                loss = prediction_loss(head, validation[LABEL_FOR_HEAD[head]].to_numpy(dtype=float), raw_score,
                                       config["targets"]["loss_quantile"])
                projected = (raw_score < -1) if head == "mu5" else ((raw_score < 0) | (raw_score > 1)) if head == "q90" else np.zeros(len(V), dtype=bool)
                scores.append({"model": model, "head": head, "parameter": float(parameter), **info, "loss": loss,
                               "projected_count": int(projected.sum()), "fit": fitted})
    selected, totals = {}, []
    for head in HEADS:
        best = float("inf")
        for parameter in sorted(config["models"][GRID_FOR_HEAD[head]], reverse=head != "p10"):
            values = [r for r in scores if r["head"] == head and r["parameter"] == parameter]
            count = sum(r["validation_rows"] for r in values)
            loss = sum(r["loss"]*r["validation_rows"] for r in values)/count
            totals.append({"model": model, "head": head, "parameter": parameter, "loss": loss, "validation_rows": count})
            if loss < best - 1e-12:
                selected[head], best = float(parameter), loss
    return selected, scores, totals, transforms
