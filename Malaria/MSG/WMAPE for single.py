import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.font_manager import FontProperties
import pandas as pd

repetitions = 10
dims = [20 * i for i in range(1, 11)]
y_Testing_set = pd.read_excel('y_Testing_set.xlsx', header=None).values

def compute_total_wmape_from_preds(preds, y_true):
    reps, n_test, k = preds.shape
    assert y_true.shape == (n_test, k), f"y_true shape {y_true.shape} 与预测 {preds.shape} 不匹配"
    denom = np.sum(np.abs(y_true))
    wmape = np.zeros(reps)
    for r in range(reps):
        abs_err = np.abs(y_true - preds[r])
        wmape[r] = np.sum(abs_err) / denom
    return wmape

WMAPE_Bagging_Meta = []
WMAPE_GP = []
WMAPE_LGB = []
WMAPE_RF = []
WMAPE_SVR = []
WMAPE_XGB = []
WMAPE_CatB = []

for dim in dims:
    P_BAG = np.load(f'PRED-MSG-{dim}.npy')
    P_GP  = np.load(f'PRED-GP-{dim}.npy')
    P_LGB = np.load(f'PRED-LGB-{dim}.npy')
    P_RF  = np.load(f'PRED-RF-{dim}.npy')
    P_SVR = np.load(f'PRED-SVR-{dim}.npy')
    P_XGB = np.load(f'PRED-XGB-{dim}.npy')
    P_Cat = np.load(f'PRED-CatBoost-{dim}.npy')

    WMAPE_Bagging_Meta.append(compute_total_wmape_from_preds(P_BAG, y_Testing_set))
    WMAPE_GP.append(compute_total_wmape_from_preds(P_GP, y_Testing_set))
    WMAPE_LGB.append(compute_total_wmape_from_preds(P_LGB, y_Testing_set))
    WMAPE_RF.append(compute_total_wmape_from_preds(P_RF, y_Testing_set))
    WMAPE_SVR.append(compute_total_wmape_from_preds(P_SVR, y_Testing_set))
    WMAPE_XGB.append(compute_total_wmape_from_preds(P_XGB, y_Testing_set))
    WMAPE_CatB.append(compute_total_wmape_from_preds(P_Cat, y_Testing_set))

def mean_std_per_dim(wmape_list):
    mean = [np.mean(w) for w in wmape_list]
    std  = [np.std(w)  for w in wmape_list]
    return np.array(mean), np.array(std)

sum_WMAPE_Bagging_Meta_mean, sum_WMAPE_Bagging_Meta_std = mean_std_per_dim(WMAPE_Bagging_Meta)
sum_WMAPE_GP_mean, sum_WMAPE_GP_std = mean_std_per_dim(WMAPE_GP)
sum_WMAPE_LGB_mean, sum_WMAPE_LGB_std = mean_std_per_dim(WMAPE_LGB)
sum_WMAPE_RF_mean, sum_WMAPE_RF_std = mean_std_per_dim(WMAPE_RF)
sum_WMAPE_SVR_mean, sum_WMAPE_SVR_std = mean_std_per_dim(WMAPE_SVR)
sum_WMAPE_XGB_mean, sum_WMAPE_XGB_std = mean_std_per_dim(WMAPE_XGB)
sum_WMAPE_CatB_mean, sum_WMAPE_CatB_std = mean_std_per_dim(WMAPE_CatB)

fig, ax = plt.subplots(figsize=(9, 8))
x = range(1, 11)

line1, = plt.plot(x, sum_WMAPE_GP_mean, linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                  label='GP', linewidth=5, zorder=8)
ax.fill_between(x, sum_WMAPE_GP_mean - 1 * sum_WMAPE_GP_std,
                sum_WMAPE_GP_mean + 1 * sum_WMAPE_GP_std, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2, zorder=1)

line2, = plt.plot(x, sum_WMAPE_SVR_mean, linestyle='-', color=mcolors.to_rgba('#CCD376'),
                  label='SVR', linewidth=5, zorder=9)
ax.fill_between(x, sum_WMAPE_SVR_mean - sum_WMAPE_SVR_std,
                sum_WMAPE_SVR_mean + sum_WMAPE_SVR_std, color=mcolors.to_rgba('#CCD376'), alpha=0.2, zorder=2)

line3, = plt.plot(x, sum_WMAPE_RF_mean, linestyle='-', color=mcolors.to_rgba('#F5AA61'),
                  label='RF', linewidth=5, zorder=11)
ax.fill_between(x, sum_WMAPE_RF_mean - sum_WMAPE_RF_std,
                sum_WMAPE_RF_mean + sum_WMAPE_RF_std, color=mcolors.to_rgba('#F5AA61'), alpha=0.2, zorder=4)

line4, = plt.plot(x, sum_WMAPE_LGB_mean, linestyle='-', color=mcolors.to_rgba('#EB7E60'),
                  label='LightGBM', linewidth=5, zorder=10)
ax.fill_between(x, sum_WMAPE_LGB_mean - sum_WMAPE_LGB_std,
                sum_WMAPE_LGB_mean + sum_WMAPE_LGB_std, color=mcolors.to_rgba('#EB7E60'), alpha=0.2, zorder=3)

line5, = plt.plot(x, sum_WMAPE_CatB_mean, linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                  label='CatBoost', linewidth=5, zorder=13)
ax.fill_between(x, sum_WMAPE_CatB_mean - sum_WMAPE_CatB_std,
                sum_WMAPE_CatB_mean + sum_WMAPE_CatB_std, color=mcolors.to_rgba('#70CDBE'), alpha=0.2, zorder=6)

line6, = plt.plot(x, sum_WMAPE_XGB_mean, linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                  label='XGBoost', linewidth=5, zorder=12)
ax.fill_between(x, sum_WMAPE_XGB_mean - sum_WMAPE_XGB_std,
                sum_WMAPE_XGB_mean + sum_WMAPE_XGB_std, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2, zorder=5)

line7, = plt.plot(x, sum_WMAPE_Bagging_Meta_mean, linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                  label='MSG', linewidth=5, zorder=14)
ax.fill_between(x, sum_WMAPE_Bagging_Meta_mean - sum_WMAPE_Bagging_Meta_std,
                sum_WMAPE_Bagging_Meta_mean + sum_WMAPE_Bagging_Meta_std, color=mcolors.to_rgba('#AC99D2'), alpha=0.2, zorder=7)

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
                   handlelength=3, handletextpad=0.8, borderaxespad=0.3, labelspacing=0.3, markerscale=1.3,
                   facecolor='none')
legend.get_frame().set_linewidth(2.5)

plt.tight_layout()
plt.savefig('MSG VS Single Malaria.svg', dpi=800)
plt.show()

