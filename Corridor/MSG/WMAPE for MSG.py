import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.font_manager import FontProperties
import pandas as pd

repetitions = 10
dims = [60 * i for i in range(1, 11)]
y_Testing_set = pd.read_excel('y_Testing_set.xlsx', header=None).values

def compute_total_wmape_from_preds(preds, y_true):
    reps, n_test, k = preds.shape
    assert y_true.shape == (n_test, k), f"y_true shape {y_true.shape} 与预测 {preds.shape} 不匹配"
    denom = np.sum(np.abs(y_true))
    wmape = np.zeros(reps)
    for r in range(reps):
        wmape[r] = np.sum(np.abs(y_true - preds[r])) / denom
    return wmape

WMAPE_MSG = []
WMAPE_MSG_CatBoost = []
WMAPE_MSG_Model = []
WMAPE_MSG_Model_Data = []

for dim in dims:
    P_XGB  = np.load(f'PRED-MSG-{dim}.npy')
    P_Cat  = np.load(f'PRED-MSG_CatBoost-{dim}.npy')
    P_Mod  = np.load(f'PRED-MSG_Model-{dim}.npy')
    P_MDat = np.load(f'PRED-MSG_Model_Data-{dim}.npy')

    WMAPE_MSG.append(compute_total_wmape_from_preds(P_XGB,  y_Testing_set))
    WMAPE_MSG_CatBoost.append(compute_total_wmape_from_preds(P_Cat,  y_Testing_set))
    WMAPE_MSG_Model.append(compute_total_wmape_from_preds(P_Mod,     y_Testing_set))
    WMAPE_MSG_Model_Data.append(compute_total_wmape_from_preds(P_MDat, y_Testing_set))

def mean_std_per_dim(wseq):
    mean = [np.mean(w) for w in wseq]
    std  = [np.std(w)  for w in wseq]
    return np.array(mean), np.array(std)

sum_WMAPE_MSG_mean, sum_WMAPE_MSG_std = mean_std_per_dim(WMAPE_MSG)
sum_WMAPE_MSG_CatBoost_mean, sum_WMAPE_MSG_CatBoost_std = mean_std_per_dim(WMAPE_MSG_CatBoost)
sum_WMAPE_MSG_Model_mean, sum_WMAPE_MSG_Model_std = mean_std_per_dim(WMAPE_MSG_Model)
sum_WMAPE_MSG_Model_Data_mean, sum_WMAPE_MSG_Model_Data_std = mean_std_per_dim(WMAPE_MSG_Model_Data)

fig, ax = plt.subplots(figsize=(9, 8))
x = range(1, 11)

line1, = plt.plot(x, sum_WMAPE_MSG_CatBoost_mean, linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                  label='Data-het CatBoost', linewidth=5, zorder=5)
ax.fill_between(x, sum_WMAPE_MSG_CatBoost_mean - sum_WMAPE_MSG_CatBoost_std,
                   sum_WMAPE_MSG_CatBoost_mean + sum_WMAPE_MSG_CatBoost_std,
                   color=mcolors.to_rgba('#70CDBE'), alpha=0.2, zorder=1)

line2, = ax.plot(x, sum_WMAPE_MSG_mean, linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='Data-het XGBoost', linewidth=5, zorder=8)
ax.fill_between(x, sum_WMAPE_MSG_mean - sum_WMAPE_MSG_std,
                   sum_WMAPE_MSG_mean + sum_WMAPE_MSG_std,
                   color=mcolors.to_rgba('#AC99D2'), alpha=0.2, zorder=4)

line3, = plt.plot(x, sum_WMAPE_MSG_Model_mean, linestyle='-', color=mcolors.to_rgba('#F8B072'),
                  label='Model-het', linewidth=5, zorder=6)
ax.fill_between(x, sum_WMAPE_MSG_Model_mean - sum_WMAPE_MSG_Model_std,
                   sum_WMAPE_MSG_Model_mean + sum_WMAPE_MSG_Model_std,
                   color=mcolors.to_rgba('#F8B072'), alpha=0.2, zorder=2)

line4, = plt.plot(x, sum_WMAPE_MSG_Model_Data_mean, linestyle='-', color=mcolors.to_rgba('#4F99C9'),
                  label='Fully-het', linewidth=5, zorder=7)
ax.fill_between(x, sum_WMAPE_MSG_Model_Data_mean - sum_WMAPE_MSG_Model_Data_std,
                   sum_WMAPE_MSG_Model_Data_mean + sum_WMAPE_MSG_Model_Data_std,
                   color=mcolors.to_rgba('#4F99C9'), alpha=0.2, zorder=3)

ax.set_xlabel('Training sample size', fontsize=28, fontname='Arial', labelpad=10)
ax.set_ylabel(r'$\mathrm{WMAPE}$', fontsize=28, fontname='Arial', labelpad=10)
ax.tick_params(axis='both', labelsize=23, length=10, width=2.5)

x_tick_positions = np.arange(1, 11, 2)
x_ticks = [rf'$\mathit{{{i}d}}$' for i in x_tick_positions]
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_ticks, ha='center', fontsize=23, fontname='Arial')
plt.setp(ax.get_yticklabels(), fontname='Arial')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

font_prop = FontProperties(family='Arial', size=28, weight='normal')
legend = ax.legend(loc='upper right', frameon=True, edgecolor='black', prop=font_prop,
                   handlelength=3, handletextpad=0.8, borderaxespad=0.9, labelspacing=0.5, markerscale=1.3)
legend.get_frame().set_linewidth(2.5)

plt.tight_layout()
plt.savefig('MSG comparison Corridor.svg', dpi=800,
            bbox_inches='tight')
plt.show()
