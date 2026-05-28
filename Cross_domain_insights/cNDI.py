import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Results")
SAVE_DIR = SCRIPT_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PREFIX = "cNDI"

EPS = 1e-12
K_DEFAULT = 5
RANDOM_STATE = 42

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================
# 1. Representative indices, scenarios and algorithms
# ============================================================

rep_indices = {
    "Malaria": [3, 8, 11, 14, 19],
    "CitySim": [0, 2, 3, 5, 6],
    "Reactor": [0, 2],
    "Aircraft": [7, 8, 9, 44, 46],
    "Songshanhu": [0, 1, 2, 3, 4, 6, 7, 21, 25],
    "WFG3": [0, 1, 2, 3],
}

scenarios = ["Malaria", "CitySim", "Reactor", "Aircraft", "Songshanhu", "WFG3"]

scenario_display_names = {
    "Malaria": "Rural malaria transmission",
    "CitySim": "Isolated urban junction",
    "Reactor": "Coiled-tube reactor",
    "Aircraft": "Human-powered aircraft",
    "Songshanhu": "Urban arterial corridor",
    "WFG3": "WFG 3",
}

full_dims = {
    "Malaria": 23,
    "CitySim": 29,
    "Reactor": 50,
    "Aircraft": 55,
    "Songshanhu": 60,
    "WFG3": 100,
}

eff_dims = {sc: len(rep_indices[sc]) for sc in scenarios}

algorithm_order = [
    "NSGA-Full",
    "MOBO-Full",
    "NSGA-Reduced",
    "MOBO-Reduced",
    "XLO-Reduced",
]


def get_display_labels_for_scenario(sc):
    d = full_dims[sc]
    r = eff_dims[sc]

    return {
        "NSGA-Full": f"NSGA ({d})",
        "MOBO-Full": f"MOBO ({d})",
        "NSGA-Reduced": f"NSGA ({r})",
        "MOBO-Reduced": f"MOBO ({r})",
        "XLO-Reduced": f"XLO ({r})",
    }


# ============================================================
# 2. Bounds for PS normalization
# ============================================================

bounds_all_Malaria = np.array([
    [0, 6],
    [0, 10000],
    [0, 14],
    [1, 75],
    [0, 1000],
    [0, 200],
    [0, 5],
    [0, 11],
    [0, 26],
    [0, 66],
    [0, 20000],
    [10000, 200000],
    [0, 2],
    [0, 9],
    [0, 1],
    [3, 651],
    [1, 16],
    [20000, 1600000],
    [0, 1],
    [0, 4],
    [0, 10],
    [0, 600],
    [0, 3],
], dtype=float)

bounds_all_CitySim = np.array([
    [0.2, 2],
    [0, 1],
    [1, 10],
    [1, 3],
    [1, 6],
    [1e-09, 3],
    [0.1, 10],
    [6, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [-1, 1],
    [-1, 10],
    [0, 1],
    [1e-09, 10],
    [0, 30],
    [0, 1],
    [0, 20 / 3.6],
    [0, 1],
    [0, 1],
    [0, 6],
    [0, 3],
    [0, 10],
], dtype=float)

bounds_all_Reactor = np.zeros((50, 2), dtype=float)
bounds_all_Reactor[:, 1] = 1.0

bounds_all_Aircraft = np.zeros((55, 2), dtype=float)
bounds_all_Aircraft[:, 1] = 1.0

bounds_all_Songshanhu = np.array([
    [3, 7],
    [0.2, 2],
    [0, 1],
    [1, 10],
    [1, 3],
    [1, 6],
    [1e-09, 3],
    [0.1, 10],
    [6, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [-1, 1],
    [-1, 10],
    [0, 1],
    [1e-09, 10],
    [0, 30],
    [0, 1],
    [0, 20 / 3.6],
    [0, 1],
    [0, 1],
    [0, 6],
    [0, 3],
    [0, 10],

    [7, 12],
    [0.2, 2],
    [0, 1],
    [1, 10],
    [1, 3],
    [1, 6],
    [1e-09, 3],
    [0.1, 10],
    [6, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 1],
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 10],
    [-1, 1],
    [-1, 10],
    [0, 1],
    [1e-09, 10],
    [0, 30],
    [0, 1],
    [0, 20 / 3.6],
    [0, 1],
    [0, 1],
    [0, 6],
    [0, 3],
    [0, 10],
], dtype=float)

bounds_all_WFG3 = np.zeros((100, 2), dtype=float)
for i in range(100):
    bounds_all_WFG3[i, 1] = 2 * (i + 1)

bounds_all = {
    "Malaria": bounds_all_Malaria,
    "CitySim": bounds_all_CitySim,
    "Reactor": bounds_all_Reactor,
    "Aircraft": bounds_all_Aircraft,
    "Songshanhu": bounds_all_Songshanhu,
    "WFG3": bounds_all_WFG3,
}


# ============================================================
# 3. File specifications
# ============================================================

def repeat_files(prefix, repetitions):
    return [f"{prefix}-repetition-{j}.npy" for j in repetitions]


algo_specs = {
    "Malaria": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA23-malaria", range(10)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO23-malaria", range(10)),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA5-malaria", range(10)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO5-malaria", range(10)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": repeat_files("PS-XLO-malaria", range(10)),
            "type": "reduced_auto",
        },
    },

    "CitySim": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA29-Junction", range(10)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO29-Junction", range(10)),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA5-Junction", range(10)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO5-Junction", range(10)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": repeat_files("PS-XLO-Junction", range(10)),
            "type": "reduced_auto",
        },
    },

    "Reactor": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA50-Reactor", range(5)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO50-Reactor", range(5)),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA2-Reactor", range(5)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO2-Reactor", range(5)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": repeat_files("PS-XLO-Reactor", range(5)),
            "type": "reduced_auto",
        },
    },

    "Aircraft": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA55-Aircraft", range(5)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO55-Aircraft", range(5)),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA5-Aircraft", range(5)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO5-Aircraft", range(5)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": repeat_files("PS-XLO-Aircraft", range(5)),
            "type": "reduced_auto",
        },
    },

    "Songshanhu": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA60-Corridor", range(10)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO60-Corridor", range(10)),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA9-Corridor", range(10)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO9-Corridor", range(10)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": repeat_files("PS-XLO-Corridor", range(10)),
            "type": "reduced_auto",
        },
    },

    "WFG3": {
        "NSGA-Full": {
            "ps_files": repeat_files("PS-NSGA100-WFG3", range(10)),
            "type": "full",
        },
        "MOBO-Full": {
            "ps_files": repeat_files("PS-MOBO100-WFG3", [0, 1, 3, 4, 5, 6, 7, 9]),
            "type": "full",
        },
        "NSGA-Reduced": {
            "ps_files": repeat_files("PS-NSGA4-WFG3", range(10)),
            "type": "reduced",
        },
        "MOBO-Reduced": {
            "ps_files": repeat_files("PS-MOBO4-WFG3", range(10)),
            "type": "reduced",
        },
        "XLO-Reduced": {
            "ps_files": [],
            "type": "reduced_auto",
        },
    },
}


def ps_to_pf_name(ps_file):
    if ps_file.startswith("PS-"):
        return "PF-" + ps_file[3:]
    return ps_file.replace("PS", "PF", 1)


for sc in scenarios:
    for alg in algorithm_order:
        algo_specs[sc][alg]["pf_files"] = [
            ps_to_pf_name(f) for f in algo_specs[sc][alg]["ps_files"]
        ]


def get_algorithms_for_scenario(sc):
    return [
        alg for alg in algorithm_order
        if alg in algo_specs[sc] and len(algo_specs[sc][alg].get("ps_files", [])) > 0
    ]


# ============================================================
# 4. Utility functions
# ============================================================

def load_npy(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    return np.load(path)


def normalize_ps_and_get_global_indices(ps_arr, scenario, search_type):
    X = np.asarray(ps_arr, dtype=float).copy()

    if X.ndim != 2:
        raise ValueError(f"{scenario}: PS must be 2D, got shape {X.shape}")

    b_full = bounds_all[scenario]
    rep = np.array(rep_indices[scenario], dtype=int)
    b_rep = b_full[rep, :]

    d = full_dims[scenario]
    k = len(rep)

    if search_type == "full":
        if X.shape[1] != d:
            raise ValueError(
                f"{scenario}: full PS expects {d} columns, got {X.shape[1]}"
            )
        X_norm = (X - b_full[:, 0]) / (b_full[:, 1] - b_full[:, 0])
        global_indices = np.arange(d, dtype=int)
        return X_norm, global_indices

    if search_type in ["reduced", "reduced_auto"]:
        if X.shape[1] == k:
            X_norm = (X - b_rep[:, 0]) / (b_rep[:, 1] - b_rep[:, 0])
            global_indices = rep.copy()
            return X_norm, global_indices

        if X.shape[1] == d:
            X_rep = X[:, rep]
            X_norm = (X_rep - b_rep[:, 0]) / (b_rep[:, 1] - b_rep[:, 0])
            global_indices = rep.copy()
            return X_norm, global_indices

        raise ValueError(
            f"{scenario}: reduced PS expects {k} or {d} columns, got {X.shape[1]}"
        )

    raise ValueError(f"Unknown search_type: {search_type}")


def normalize_pf_with_bounds(pf_arr, lower, upper):
    Y = np.asarray(pf_arr, dtype=float).copy()

    if Y.ndim != 2:
        raise ValueError(f"PF must be 2D, got shape {Y.shape}")

    Y_norm = (Y - lower) / (upper - lower + EPS)
    return np.clip(Y_norm, 0.0, 1.0)


def choose_keff(n_pareto, m, k_default=5):
    n_min = max(5, 2 * m)
    k_eff = min(k_default, int(np.floor(n_pareto / n_min)))

    if k_eff < 2:
        k_eff = 1

    return int(k_eff), int(n_min)


def get_kmeans_labels(Y_norm, k_default=5, random_state=42):
    Y_norm = np.asarray(Y_norm, dtype=float)

    if Y_norm.ndim != 2:
        raise ValueError(f"Y_norm must be 2D, got shape {Y_norm.shape}")

    n_pareto, m = Y_norm.shape
    k_eff, n_min = choose_keff(n_pareto, m, k_default=k_default)

    if k_eff <= 1:
        labels = np.zeros(n_pareto, dtype=int)
        return labels, k_eff, n_min

    unique_y = np.unique(Y_norm, axis=0)
    if unique_y.shape[0] < k_eff:
        k_eff = unique_y.shape[0]

    if k_eff <= 1:
        labels = np.zeros(n_pareto, dtype=int)
        return labels, k_eff, n_min

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = KMeans(
            n_clusters=k_eff,
            n_init=20,
            random_state=random_state,
        ).fit_predict(Y_norm)

    return labels.astype(int), k_eff, n_min


def compute_cndi(X_norm, labels):
    X = np.asarray(X_norm, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X and labels must have the same number of rows: "
            f"{X.shape[0]} vs {labels.shape[0]}"
        )

    n, d = X.shape

    if n < 2:
        return np.full(d, np.nan), np.full(d, np.nan)

    within_var = np.zeros(d, dtype=float)

    for lab in np.unique(labels):
        idx = labels == lab
        nk = int(np.sum(idx))

        if nk == 0:
            continue

        weight = nk / n
        Xk = X[idx, :]
        var_k = np.var(Xk, axis=0, ddof=0)
        within_var += weight * var_k

    within_var = np.maximum(within_var, 0.0)
    cndi = np.sqrt(12.0 * within_var)

    return cndi, within_var


def significance_marker(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


# ============================================================
# 5. Read raw PS and PF files run-wise
# ============================================================

PS_RAW = {}
PF_RAW = {}

for sc in scenarios:
    PS_RAW[sc] = {}
    PF_RAW[sc] = {}

    for alg in get_algorithms_for_scenario(sc):
        spec = algo_specs[sc][alg]
        ps_files = spec["ps_files"]
        pf_files = spec["pf_files"]

        if len(ps_files) != len(pf_files):
            raise ValueError(f"{sc}-{alg}: PS and PF file lists have different lengths.")

        PS_RAW[sc][alg] = []
        PF_RAW[sc][alg] = []

        for run_id, (ps_file, pf_file) in enumerate(zip(ps_files, pf_files)):
            ps = load_npy(ps_file)
            pf = load_npy(pf_file)

            if ps.shape[0] != pf.shape[0]:
                raise ValueError(
                    f"{sc}-{alg}-run{run_id}: PS and PF row mismatch: "
                    f"PS {ps.shape}, PF {pf.shape}"
                )

            PS_RAW[sc][alg].append(ps)
            PF_RAW[sc][alg].append(pf)

print("Finished reading raw PS and PF files.")


# ============================================================
# 6. Scenario-level pooled min-max normalization bounds for PF
# ============================================================

pf_norm_bounds = {}

for sc in scenarios:
    pf_pool = []
    m_set = set()

    for alg in get_algorithms_for_scenario(sc):
        for pf in PF_RAW[sc][alg]:
            pf = np.asarray(pf, dtype=float)
            pf_pool.append(pf)
            m_set.add(pf.shape[1])

    if len(m_set) != 1:
        raise ValueError(f"{sc}: inconsistent objective dimensions in PF files: {m_set}")

    pf_pool = np.vstack(pf_pool)
    lower = np.min(pf_pool, axis=0)
    upper = np.max(pf_pool, axis=0)

    pf_norm_bounds[sc] = {
        "lower": lower,
        "upper": upper,
        "m": pf_pool.shape[1],
    }

print("Finished computing scenario-level PF normalization bounds.")


# ============================================================
# 7. Compute cNDI only
# ============================================================

rows = []
run_info_rows = []

for sc in scenarios:
    rep = set(rep_indices[sc])
    lower = pf_norm_bounds[sc]["lower"]
    upper = pf_norm_bounds[sc]["upper"]

    for alg in get_algorithms_for_scenario(sc):
        spec = algo_specs[sc][alg]
        search_type = spec["type"]

        for run_id, (ps_raw, pf_raw) in enumerate(zip(PS_RAW[sc][alg], PF_RAW[sc][alg])):
            X_norm, global_indices = normalize_ps_and_get_global_indices(
                ps_raw,
                scenario=sc,
                search_type=search_type,
            )

            X_norm = np.clip(X_norm, 0.0, 1.0)
            Y_norm = normalize_pf_with_bounds(pf_raw, lower=lower, upper=upper)

            labels, k_eff, n_min = get_kmeans_labels(
                Y_norm,
                k_default=K_DEFAULT,
                random_state=RANDOM_STATE,
            )

            cndi, within_var = compute_cndi(X_norm, labels)

            run_info_rows.append({
                "Scenario": sc,
                "Scenario_Display": scenario_display_names[sc],
                "Algorithm": alg,
                "Run": run_id,
                "Num_Pareto": X_norm.shape[0],
                "Num_Objectives": Y_norm.shape[1],
                "n_min": n_min,
                "K_eff": k_eff,
                "Num_Parameters_In_File": X_norm.shape[1],
                "Is_Full_Dim_File": X_norm.shape[1] == full_dims[sc],
            })

            for local_idx, global_idx in enumerate(global_indices):
                parameter_type = "Representative" if int(global_idx) in rep else "Redundant"

                rows.append({
                    "Scenario": sc,
                    "Scenario_Display": scenario_display_names[sc],
                    "Algorithm": alg,
                    "Run": run_id,
                    "Local_Parameter_Index": int(local_idx),
                    "Global_Parameter_Index": int(global_idx),
                    "Parameter_Type": parameter_type,
                    "cNDI": float(cndi[local_idx]),
                    "Within_Niche_Variance": float(within_var[local_idx]),
                    "Num_Pareto": X_norm.shape[0],
                    "K_eff": k_eff,
                })

metrics_df = pd.DataFrame(rows)
run_info_df = pd.DataFrame(run_info_rows)

metrics_csv = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_raw_cNDI.csv")
run_info_csv = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_run_info.csv")

metrics_df.to_csv(metrics_csv, index=False)
run_info_df.to_csv(run_info_csv, index=False)

print(f"Saved raw cNDI metrics to: {metrics_csv}")
print(f"Saved run information to: {run_info_csv}")


# ============================================================
# 8. Summary statistics and representative-vs-redundant tests
# ============================================================

summary_df = (
    metrics_df
    .groupby(["Scenario", "Scenario_Display", "Algorithm", "Parameter_Type"])
    .agg(
        N=("cNDI", "size"),
        Median_cNDI=("cNDI", "median"),
        Mean_cNDI=("cNDI", "mean"),
        Std_cNDI=("cNDI", "std"),
    )
    .reset_index()
)

summary_csv = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved cNDI summary to: {summary_csv}")

test_rows = []

for sc in scenarios:
    for alg in get_algorithms_for_scenario(sc):
        sub = metrics_df[(metrics_df["Scenario"] == sc) & (metrics_df["Algorithm"] == alg)]

        rep_vals = sub[sub["Parameter_Type"] == "Representative"]["cNDI"].dropna().values
        red_vals = sub[sub["Parameter_Type"] == "Redundant"]["cNDI"].dropna().values

        if len(rep_vals) == 0 or len(red_vals) == 0:
            continue

        try:
            stat, pval = mannwhitneyu(rep_vals, red_vals, alternative="less")
        except ValueError:
            stat, pval = np.nan, np.nan

        med_rep = np.median(rep_vals)
        med_red = np.median(red_vals)
        ratio = med_red / (med_rep + EPS)

        test_rows.append({
            "Scenario": sc,
            "Scenario_Display": scenario_display_names[sc],
            "Algorithm": alg,
            "N_Representative": len(rep_vals),
            "N_Redundant": len(red_vals),
            "Median_cNDI_Representative": med_rep,
            "Median_cNDI_Redundant": med_red,
            "Redundant_over_Representative_Median_Ratio": ratio,
            "MannWhitneyU_stat": stat,
            "P_value_Representative_less_Redundant": pval,
            "Significance": significance_marker(pval),
        })

test_df = pd.DataFrame(test_rows)
test_csv = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_rep_vs_red_tests.csv")
test_df.to_csv(test_csv, index=False)
print(f"Saved representative-vs-redundant tests to: {test_csv}")


# ============================================================
# 9. Plotting functions: boxplot + half violin + scatter
# ============================================================

COLORS = {
    "Representative": "#4C72B0",
    "Redundant": "#C44E52",
}


def half_violin(ax, data, pos, side="left", color="#4C72B0", width=0.55, alpha=0.28):
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return

    if len(data) == 1 or np.allclose(data, data[0]):
        rng = np.random.default_rng(123)
        x = np.full(len(data), pos) + rng.normal(0, 0.015, size=len(data))
        ax.scatter(
            x,
            data,
            s=16,
            color=color,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.25,
            zorder=5,
        )
        return

    parts = ax.violinplot(
        data,
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)

        verts = body.get_paths()[0].vertices

        if side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], pos)
        elif side == "right":
            verts[:, 0] = np.maximum(verts[:, 0], pos)


def box_and_scatter(ax, data, pos, color="#4C72B0", box_width=0.15, jitter=0.035, seed=0):
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return

    ax.boxplot(
        data,
        positions=[pos],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.25),
        boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.25),
        whiskerprops=dict(color=color, linewidth=1.0),
        capprops=dict(color=color, linewidth=1.0),
    )

    rng = np.random.default_rng(seed)
    x = np.full(len(data), pos) + rng.normal(0, jitter, size=len(data))

    ax.scatter(
        x,
        data,
        s=18,
        color=color,
        alpha=0.58,
        edgecolors="black",
        linewidths=0.4,
        zorder=4,
    )


def get_group_data_top(rep_vals, red_vals=None):
    vals = []

    rep_vals = np.asarray(rep_vals, dtype=float).ravel()
    rep_vals = rep_vals[np.isfinite(rep_vals)]
    if len(rep_vals) > 0:
        vals.append(np.max(rep_vals))

    if red_vals is not None:
        red_vals = np.asarray(red_vals, dtype=float).ravel()
        red_vals = red_vals[np.isfinite(red_vals)]
        if len(red_vals) > 0:
            vals.append(np.max(red_vals))

    if len(vals) == 0:
        return 0.0

    return float(np.max(vals))


def draw_cndi_figure(df, test_df):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18.8, 9.8),
        sharey=False,
    )
    axes = axes.flatten()

    panel_label_x = -0.12
    panel_label_y = 1.06
    ylabel_x = -0.1

    title_fontsize = 15.5
    ylabel_fontsize = 14
    tick_fontsize = 11.5
    annotation_fontsize = 11.5
    panel_label_fontsize = 18

    for ax_id, sc in enumerate(scenarios):
        ax = axes[ax_id]
        sub_sc = df[df["Scenario"] == sc]
        display_labels = get_display_labels_for_scenario(sc)
        algs_sc = get_algorithms_for_scenario(sc)

        subplot_data_max = 0.0
        annotation_items = []

        for x_id, alg in enumerate(algs_sc, start=1):
            sub_alg = sub_sc[sub_sc["Algorithm"] == alg]

            rep_vals = sub_alg[sub_alg["Parameter_Type"] == "Representative"]["cNDI"].dropna().values
            red_vals = sub_alg[sub_alg["Parameter_Type"] == "Redundant"]["cNDI"].dropna().values

            group_top = get_group_data_top(rep_vals, red_vals if len(red_vals) > 0 else None)
            subplot_data_max = max(subplot_data_max, group_top)

            if len(red_vals) > 0:
                test_row = test_df[
                    (test_df["Scenario"] == sc) &
                    (test_df["Algorithm"] == alg)
                ]

                if len(test_row) > 0:
                    ratio = float(test_row["Redundant_over_Representative_Median_Ratio"].iloc[0])
                    pval = float(test_row["P_value_Representative_less_Redundant"].iloc[0])

                    annotation_items.append({
                        "x": x_id,
                        "group_top": group_top,
                        "ratio": ratio,
                        "pval": pval,
                    })

        subplot_data_max = max(subplot_data_max, 1.0)

        offset = max(0.025, 0.035 * subplot_data_max)

        for item in annotation_items:
            item["y"] = item["group_top"] + offset

        annotation_top = max([item["y"] for item in annotation_items], default=0.0)

        y_max = max(subplot_data_max, annotation_top)
        y_max = y_max + max(0.06, 0.075 * y_max)
        y_max = max(1.12, y_max)
        y_max = np.ceil(y_max * 10) / 10.0
        y_min = -0.05

        for x_id, alg in enumerate(algs_sc, start=1):
            sub_alg = sub_sc[sub_sc["Algorithm"] == alg]

            rep_vals = sub_alg[sub_alg["Parameter_Type"] == "Representative"]["cNDI"].dropna().values
            red_vals = sub_alg[sub_alg["Parameter_Type"] == "Redundant"]["cNDI"].dropna().values

            if len(red_vals) > 0:
                pos_rep = x_id - 0.16
                pos_red = x_id + 0.16

                half_violin(
                    ax,
                    rep_vals,
                    pos_rep,
                    side="left",
                    color=COLORS["Representative"],
                    width=0.50,
                    alpha=0.32,
                )

                half_violin(
                    ax,
                    red_vals,
                    pos_red,
                    side="right",
                    color=COLORS["Redundant"],
                    width=0.50,
                    alpha=0.32,
                )

                box_and_scatter(
                    ax,
                    rep_vals,
                    pos_rep,
                    color=COLORS["Representative"],
                    seed=1000 + ax_id * 100 + x_id,
                )

                box_and_scatter(
                    ax,
                    red_vals,
                    pos_red,
                    color=COLORS["Redundant"],
                    seed=2000 + ax_id * 100 + x_id,
                )

            else:
                pos_rep = x_id

                half_violin(
                    ax,
                    rep_vals,
                    pos_rep,
                    side="left",
                    color=COLORS["Representative"],
                    width=0.55,
                    alpha=0.32,
                )

                box_and_scatter(
                    ax,
                    rep_vals,
                    pos_rep,
                    color=COLORS["Representative"],
                    seed=3000 + ax_id * 100 + x_id,
                )

        for item in annotation_items:
            sig = significance_marker(item["pval"])
            text = f"{item['ratio']:.2f}×\n{sig}"

            ax.text(
                item["x"],
                item["y"],
                text,
                ha="center",
                va="bottom",
                fontsize=annotation_fontsize,
                color="black",
                linespacing=0.95,
                zorder=10,
            )

        ax.axhline(
            1.0,
            color="gray",
            linestyle="--",
            linewidth=1.25,
            alpha=0.75,
            zorder=0,
        )

        ax.set_ylim(y_min, y_max)

        ax.set_title(
            scenario_display_names[sc],
            fontsize=title_fontsize,
            fontweight="bold",
            pad=10,
        )

        ax.set_xticks(range(1, len(algs_sc) + 1))
        ax.set_xticklabels(
            [display_labels[a] for a in algs_sc],
            fontsize=tick_fontsize,
            rotation=0,
            ha="center",
            fontname="Arial",
        )

        # 每个子图都保留自己的纵轴 label
        ax.set_ylabel("cNDI", fontsize=ylabel_fontsize)
        ax.yaxis.set_label_coords(ylabel_x, 0.5)

        ax.text(
            panel_label_x,
            panel_label_y,
            "abcdef"[ax_id],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            fontname="Arial",
            clip_on=False,
            zorder=30,
        )

        ax.grid(
            axis="y",
            alpha=0.22,
            linewidth=0.9,
        )

        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
            length=4.5,
            width=1.2,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.25)
        ax.spines["bottom"].set_linewidth(1.25)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["Representative"],
            markeredgecolor="black",
            markersize=9.5,
            label="Representative factors",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["Redundant"],
            markeredgecolor="black",
            markersize=9.5,
            label="Weakly contributing factors",
        ),
        plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Uniform-drift reference",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=13.5,
        handlelength=1.8,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.875,
        wspace=0.18,
        hspace=0.26,
    )

    fig.savefig(
        os.path.join(SAVE_DIR, f"{SAVE_PREFIX}.svg"),
        bbox_inches="tight",
        pad_inches=0.04,
    )

    return fig


# ============================================================
# 10. Draw cNDI figure only
# ============================================================

fig_cndi = draw_cndi_figure(metrics_df, test_df)
plt.close(fig_cndi)


# ============================================================
# 11. Diagnostics
# ============================================================

print("\n=== PF normalization bounds ===")
for sc in scenarios:
    print(sc)
    print("lower:", pf_norm_bounds[sc]["lower"])
    print("upper:", pf_norm_bounds[sc]["upper"])
    print("m:", pf_norm_bounds[sc]["m"])

print("\n=== K_eff distribution ===")
print(
    run_info_df
    .groupby(["Scenario", "Algorithm"])["K_eff"]
    .describe()
)

print("\n=== cNDI summary ===")
print(summary_df)

print("\n=== Representative-vs-redundant cNDI tests ===")
print(test_df)

print("\nAll tasks completed.")