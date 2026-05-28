import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

PF_NSGA29 = []
for j in range(10):
    PF_NSGA29.append(np.load(os.path.join(root_dir,
        'NSGA29\Results\PF-NSGA29-Junction-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA29 = np.array(np.concatenate(PF_NSGA29, axis=0))

PF_MOBO29 = []
for j in range(10):
    PF_MOBO29.append(np.load(os.path.join(root_dir,
        'MOBO29\Results\PF-MOBO29-Junction-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO29 = np.array(np.concatenate(PF_MOBO29, axis=0))

PF_NSGA5 = []
for j in range(10):
    PF_NSGA5.append(np.load(os.path.join(root_dir,
        'NSGA5\Results\PF-NSGA5-Junction-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA5 = np.concatenate(PF_NSGA5, axis=0)

PF_MOBO5 = []
for j in range(10):
    PF_MOBO5.append(np.load(os.path.join(root_dir,
        'MOBO5\Results\PF-MOBO5-Junction-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO5 = np.concatenate(PF_MOBO5, axis=0)

PF_XLO5 = []
for j in range(10):
    PF_XLO5.append(np.load(os.path.join(root_dir,
        'XLO\Results\PF-XLO-Junction-repetition-' + str(
            j) + '.npy')))

all_PF_XLO5 = np.concatenate(PF_XLO5, axis=0)

all_PF_NSGA29 = np.unique(all_PF_NSGA29, axis=0)
all_PF_MOBO29 = np.unique(all_PF_MOBO29, axis=0)
all_PF_NSGA5 = np.unique(all_PF_NSGA5, axis=0)
all_PF_MOBO5 = np.unique(all_PF_MOBO5, axis=0)
all_PF_XLO5 = np.unique(all_PF_XLO5, axis=0)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

norm = plt.Normalize(
    vmin=min(min(all_PF_NSGA29[:, -1]), min(all_PF_MOBO29[:, -1]), min(all_PF_NSGA5[:, -1]), min(all_PF_MOBO5[:, -1]),
             min(all_PF_XLO5[:, -1])),
    vmax=max(max(all_PF_NSGA29[:, -1]), max(all_PF_MOBO29[:, -1]), max(all_PF_NSGA5[:, -1]),
             max(all_PF_MOBO5[:, -1]), max(all_PF_XLO5[:, -1])))

color_list = ["#FFDD8E", "#70CDBE", "#8FB4DC", "#AC99D2"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", color_list, N=256)

sc1 = ax.scatter(all_PF_NSGA29[:, 0], all_PF_NSGA29[:, 1], all_PF_NSGA29[:, 2],
                 c=all_PF_NSGA29[:, -1], cmap=cmap,
                 marker='^', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='NSGA-II (29)', zorder=1)

sc2 = ax.scatter(all_PF_MOBO29[:, 0], all_PF_MOBO29[:, 1], all_PF_MOBO29[:, 2],
                 c=all_PF_MOBO29[:, -1], cmap=cmap,
                 marker='o', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='MOBO (29)', zorder=2)

sc3 = ax.scatter(all_PF_NSGA5[:, 0], all_PF_NSGA5[:, 1], all_PF_NSGA5[:, 2],
                 c=all_PF_NSGA5[:, -1], cmap=cmap,
                 marker='v', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='NSGA-II (5)', zorder=3)

sc4 = ax.scatter(all_PF_MOBO5[:, 0], all_PF_MOBO5[:, 1], all_PF_MOBO5[:, 2],
                 c=all_PF_MOBO5[:, -1], cmap=cmap,
                 marker='s', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='MOBO (5)', zorder=4)

sc5 = ax.scatter(all_PF_XLO5[:, 0], all_PF_XLO5[:, 1], all_PF_XLO5[:, 2],
                 c=all_PF_XLO5[:, -1], cmap=cmap,
                 marker='*', s=190, edgecolor='k', alpha=1, linewidths=0.7, label='XLO (5)', zorder=5)

ax.view_init(elev=20, azim=235)

ax.set_xlabel(r'$\mathrm{WMAPE(\mathbf{v})}$', labelpad=15, fontsize=23, fontname='Arial')
ax.set_ylabel(r'$\mathrm{WMAPE(\mathbf{t})}$', labelpad=15, fontsize=23, fontname='Arial')

ax.set_zlabel('', labelpad=8)
ax.text2D(
    -0.052, 0.5,
    r'$\mathrm{WMAPE(\mathbf{l})}$',
    transform=ax.transAxes,
    fontsize=23,
    fontname='Arial',
    rotation=91,
    va='center',
    ha='center'
)
ax.tick_params(axis='both', labelsize=18, length=100, width=2.5)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=8)
ax.xaxis.set_major_locator(MultipleLocator(0.02))

font_prop = FontProperties(
    family='Arial',
    size=23,
    weight='normal'
)

legend_items = [
    Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=15, label='NSGA-II (29)'),
    Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=15, label='MOBO (29)'),
    Line2D([0], [0], marker='v', color='black', linestyle='None', markersize=15, label='NSGA-II (5)'),
    Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=15, label='MOBO (5)'),
    Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=20, label='XLO (5)')
]

legend = ax.legend(
    handles=legend_items,
    bbox_to_anchor=(1.0, 0.88),
    frameon=True,
    edgecolor='black',
    prop=font_prop,
    handletextpad=0.3,
    borderaxespad=0.3,
    labelspacing=0.3,
    markerscale=1.1
)
legend.get_frame().set_linewidth(2.5)

fig.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)

cbar = plt.colorbar(sc5, ax=ax, shrink=0.45, aspect=8, pad=0.02)
cbar.outline.set_linewidth(1)
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
cb_tick_font = FontProperties(family='Arial', size=18, weight='normal')
cbar.ax.tick_params(width=2, length=10)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontproperties(cb_tick_font)
cbar.set_label(r'$\mathrm{WMAPE(\mathbf{q})}$', labelpad=-50, y=-0.05, rotation=360, fontsize=25)
cbar.ax.set_position([0.79, 0.18, 0.9, 0.29])

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.savefig('PF Junction.svg', dpi=800)

plt.show()