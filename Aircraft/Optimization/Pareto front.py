import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import mpl_toolkits.mplot3d.axis3d as axis3d
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

PF_NSGA55 = []
for j in range(5):
    PF_NSGA55.append(np.load(os.path.join(root_dir,
        'NSGA55\Results\PF-NSGA55-Aircraft-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA55 = np.array(np.concatenate(PF_NSGA55, axis=0))

PF_MOBO55 = []
for j in range(5):
    PF_MOBO55.append(np.load(os.path.join(root_dir,
        'MOBO55\Results\PF-MOBO55-Aircraft-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO55 = np.array(np.concatenate(PF_MOBO55, axis=0))

PF_NSGA5 = []
for j in range(5):
    PF_NSGA5.append(np.load(os.path.join(root_dir,
        'NSGA5\Results\PF-NSGA5-Aircraft-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA5 = np.concatenate(PF_NSGA5, axis=0)

PF_MOBO5 = []
for j in range(5):
    PF_MOBO5.append(np.load(os.path.join(root_dir,
        'MOBO5\Results\PF-MOBO5-Aircraft-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO5 = np.concatenate(PF_MOBO5, axis=0)

PF_XLO5 = []
for j in range(5):
    PF_XLO5.append(np.load(os.path.join(root_dir,
        'XLO\Results\PF-XLO-Aircraft-repetition-' + str(
            j) + '.npy')))

all_PF_XLO5 = np.concatenate(PF_XLO5, axis=0)

all_PF_NSGA55 = np.unique(all_PF_NSGA55, axis=0)
all_PF_MOBO55 = np.unique(all_PF_MOBO55, axis=0)
all_PF_NSGA5 = np.unique(all_PF_NSGA5, axis=0)
all_PF_MOBO5 = np.unique(all_PF_MOBO5, axis=0)
all_PF_XLO5 = np.unique(all_PF_XLO5, axis=0)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc1 = ax.scatter(all_PF_NSGA55[:, 0], all_PF_NSGA55[:, 2], all_PF_NSGA55[:, 1],
                 color=mcolors.to_rgba('#70CDBE'),
                 marker='^', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='NSGA-II (55)', zorder=1)

sc2 = ax.scatter(all_PF_MOBO55[:, 0], all_PF_MOBO55[:, 2], all_PF_MOBO55[:, 1],
                 color=mcolors.to_rgba('#F9B3AD'),
                 marker='o', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='MOBO (55)', zorder=2)

sc3 = ax.scatter(all_PF_NSGA5[:, 0], all_PF_NSGA5[:, 2], all_PF_NSGA5[:, 1],
                 color=mcolors.to_rgba('#FFDD8E'),
                 marker='v', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='NSGA-II (5)', zorder=3)

sc4 = ax.scatter(all_PF_MOBO5[:, 0], all_PF_MOBO5[:, 2], all_PF_MOBO5[:, 1],
                 color=mcolors.to_rgba('#8FB4DC'),
                 marker='s', s=100, edgecolor='k', alpha=1, linewidths=0.7, label='MOBO (5)', zorder=4)

sc5 = ax.scatter(all_PF_XLO5[:, 0], all_PF_XLO5[:, 2], all_PF_XLO5[:, 1],
                 color=mcolors.to_rgba('#AC99D2'),
                 marker='*', s=190, edgecolor='k', alpha=1, linewidths=0.7, label='XLO (5)', zorder=5)


ax.view_init(elev=20, azim=295)

ax.set_xlabel(r'$D$', labelpad=15, fontsize=23, fontname='Arial')
ax.set_ylabel(r'$W$', labelpad=15, fontsize=23, fontname='Arial')
ax.set_zlabel('', labelpad=8)
ax.text2D(
    -0.048, 0.44,
    r'$\alpha$',
    transform=ax.transAxes,
    fontsize=23,
    fontname='Arial',
    rotation=91,
    va='center',
    ha='center'
)
ax.zaxis._axinfo['juggled'] = (1, 2, 0)
axis3d.Axis._AXINFO['z']['tickdir'] = 1
ax.tick_params(axis='both', labelsize=18, length=100, width=2.5)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=8)

font_prop = FontProperties(
    family='Arial',
    size=23,
    weight='normal'
)


legend_items = [
    Line2D([0], [0], marker='^', color=mcolors.to_rgba('#70CDBE'),
                  linestyle='None', markersize=15, label='NSGA-II (55)',
                  markeredgecolor='black',
                  markeredgewidth=0.8),
    Line2D([0], [0], marker='o', color=mcolors.to_rgba('#F9B3AD'),
                  linestyle='None', markersize=15, label='MOBO (55)',
                  markeredgecolor='black',
                  markeredgewidth=0.8),
    Line2D([0], [0], marker='v', color=mcolors.to_rgba('#FFDD8E'),
                  linestyle='None', markersize=15, label='NSGA-II (5)',
                  markeredgecolor='black',
                  markeredgewidth=0.8),
    Line2D([0], [0], marker='s', color=mcolors.to_rgba('#8FB4DC'),
                  linestyle='None', markersize=15, label='MOBO (5)',
                  markeredgecolor='black',
                  markeredgewidth=0.8),
    Line2D([0], [0], marker='*', color=mcolors.to_rgba('#AC99D2'),
                  linestyle='None', markersize=20, label='XLO (5)',
                  markeredgecolor='black',
                  markeredgewidth=0.8)
]

legend = ax.legend(
    handles=legend_items,
    bbox_to_anchor=(1.41, 0.63),
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

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

fig.subplots_adjust(right=0.7)

plt.savefig('PF Aircraft.svg', dpi=800)

plt.show()