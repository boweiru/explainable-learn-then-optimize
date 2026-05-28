import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from deap import base, creator, tools
import mpl_toolkits.mplot3d.axis3d as axis3d
import os
from matplotlib.ticker import MaxNLocator

root_dir = os.path.dirname(os.path.abspath(__file__))

def nsga2_elite_selection(F, k):
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * F.shape[1])
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    pop = []
    for i in range(len(F)):
        ind = creator.Individual([])
        ind.fitness.values = tuple(F[i])
        pop.append(ind)

    elite = tools.selNSGA2(pop, k=k)
    elite_F = np.array([ind.fitness.values for ind in elite])
    return elite_F

PF_NSGA100 = []
for j in range(10):
    PF_NSGA100.append(np.load(os.path.join(root_dir,'NSGA100\Results\PF-NSGA100-WFG3-repetition-' + str(j) + '.npy')))

all_PF_NSGA100 = np.concatenate(PF_NSGA100, axis=0)

PF_MOBO100 = []
for i in [0, 1, 3, 4, 5, 6, 7, 9]:
    PF_MOBO100.append(np.load(os.path.join(root_dir,'MOBO100\Results\PF-MOBO100-WFG3-repetition-' + str(i) + '.npy')))

all_PF_MOBO100 = np.concatenate(PF_MOBO100, axis=0)

PF_NSGA4 = []
for j in range(10):
    PF_NSGA4.append(np.load(os.path.join(root_dir,'NSGA4\Results\PF-NSGA4-WFG3-repetition-' + str(j) + '.npy')))

all_PF_NSGA4 = np.concatenate(PF_NSGA4, axis=0)

PF_MOBO4 = []
for i in range(10):
    PF_MOBO4.append(np.load('MOBO4\Results\PF-MOBO4-WFG3-repetition-' + str(i) + '.npy'))
all_PF_MOBO4 = np.concatenate(PF_MOBO4, axis=0)

PF_XLO4 = []
for i in range(10):
    PF_XLO4.append(np.load('XLO\Results\PF-XLO-WFG3-repetition-' + str(i) + '.npy'))
all_PF_XLO4 = np.concatenate(PF_XLO4, axis=0)

all_PF_NSGA100 = np.unique(all_PF_NSGA100, axis=0)
all_PF_NSGA4 = np.unique(all_PF_NSGA4, axis=0)
all_PF_MOBO100 = np.unique(all_PF_MOBO100, axis=0)
all_PF_MOBO4 = np.unique(all_PF_MOBO4, axis=0)
all_PF_XLO4 = np.unique(all_PF_XLO4, axis=0)

all_PF_MOBO100 = nsga2_elite_selection(all_PF_MOBO100, 100)
all_PF_NSGA100 = nsga2_elite_selection(all_PF_NSGA100, 100)
all_PF_MOBO4 = nsga2_elite_selection(all_PF_MOBO4, 100)
all_PF_XLO4 = nsga2_elite_selection(all_PF_XLO4, 100)

fig = plt.figure(figsize=(12, 10))
fig.subplots_adjust(right=0.8)
ax = fig.add_subplot(111, projection='3d')

norm = plt.Normalize(
    vmin=min(min(all_PF_NSGA100[:, -1]), min(all_PF_NSGA4[:, -1]), min(all_PF_MOBO4[:, -1]), min(all_PF_XLO4[:, -1])),
    vmax=max(max(all_PF_NSGA100[:, -1]), max(all_PF_NSGA4[:, -1]), max(all_PF_MOBO4[:, -1]), max(all_PF_XLO4[:, -1])))
color_list = ["#FFDD8E", "#70CDBE", "#8FB4DC", "#AC99D2"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", color_list, N=256)

all_sizes = np.hstack(
    (all_PF_NSGA100[:, 4], all_PF_NSGA4[:, 4], all_PF_MOBO100[:, 4], all_PF_MOBO4[:, 4], all_PF_XLO4[:, 4]))
size_scale = 75
sc1 = ax.scatter(all_PF_NSGA100[:, 0], all_PF_NSGA100[:, 1], all_PF_NSGA100[:, 2],
                 c=all_PF_NSGA100[:, 3], cmap=cmap,
                 marker='^', s=size_scale * all_PF_NSGA100[:, 4], edgecolor='k', alpha=1, linewidths=0.5,
                 label='NSGA-II (100)', zorder=1)

sc2 = ax.scatter(all_PF_MOBO100[:, 0], all_PF_MOBO100[:, 1], all_PF_MOBO100[:, 2],
                 c=all_PF_MOBO100[:, 3], cmap=cmap,
                 marker='o', s=size_scale * all_PF_MOBO100[:, 4], edgecolor='k', alpha=1, linewidths=0.5,
                 label='MOBO (100)', zorder=2)

sc3 = ax.scatter(all_PF_NSGA4[:, 0], all_PF_NSGA4[:, 1], all_PF_NSGA4[:, 2],
                 c=all_PF_NSGA4[:, 3], cmap=cmap,
                 marker='v', s=size_scale * all_PF_NSGA4[:, 4], edgecolor='k', alpha=1, linewidths=0.5,
                 label='NSGA-II (4)', zorder=3)

sc4 = ax.scatter(all_PF_MOBO4[:, 0], all_PF_MOBO4[:, 1], all_PF_MOBO4[:, 2],
                 c=all_PF_MOBO4[:, 3], cmap=cmap,
                 marker='s', s=size_scale * all_PF_MOBO4[:, 4], edgecolor='k', alpha=1, linewidths=0.5,
                 label='MOBO (4)', zorder=4)

sc5 = ax.scatter(all_PF_XLO4[:, 0], all_PF_XLO4[:, 1], all_PF_XLO4[:, 2],
                 c=all_PF_XLO4[:, 3], cmap=cmap,
                 marker='*', s=size_scale * all_PF_XLO4[:, 4], edgecolor='k', alpha=1, linewidths=0.5,
                 label='XLO (4)', zorder=5)

ax.view_init(elev=20, azim=300)

ax.set_xlabel('Objective 1', labelpad=12, fontsize=23, fontname='Arial')
ax.set_ylabel('Objective 2', labelpad=12, fontsize=23, fontname='Arial')
ax.text2D(
    -0.018, 0.47,
    'Objective 3',
    transform=ax.transAxes,
    fontsize=23,
    fontname='Arial',
    rotation=91,
    va='center',
    ha='center'
)
ax.zaxis._axinfo['juggled'] = (1, 2, 0)
axis3d.Axis._AXINFO['z']['tickdir'] = 1

ax.tick_params(axis='both', labelsize=18, length=6, width=2.5)
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=3)

font_prop = FontProperties(
    family='Arial',
    size=23,
    weight='normal'
)

legend_items = [
    Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=15, label='NSGA-II (100)'),
    Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=15, label='MOBO (100)'),
    Line2D([0], [0], marker='v', color='black', linestyle='None', markersize=15, label='NSGA-II (4)'),
    Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=15, label='MOBO (4)'),
    Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=20, label='XLO (4)')
]

legend = ax.legend(
    handles=legend_items,
    bbox_to_anchor=(1.47, 0.85),
    frameon=True,
    edgecolor='black',
    prop=font_prop,
    handletextpad=0.3,
    borderaxespad=0.3,
    labelspacing=0.3,
    markerscale=1.1
)
legend.get_frame().set_linewidth(2.5)
ax.add_artist(legend)

cbar = plt.colorbar(sc1, ax=ax, shrink=0.5, aspect=8, pad=0.02)
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
cbar.outline.set_linewidth(1)
cb_tick_font = FontProperties(family='Arial', size=18, weight='normal')
cb_label_font = FontProperties(family='Arial', size=23, weight='normal')
cbar.ax.tick_params(width=2, length=10)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontproperties(cb_tick_font)
cbar.ax.set_ylabel('Objective\n4',
                   fontproperties=cb_label_font, rotation=0,
                   labelpad=-50)
cbar.ax.yaxis.set_label_coords(0.75, -0.05)
cbar.ax.set_position([0.73, 0.24, 0.4, 0.25])

legend_ax = fig.add_axes([0.75, 0.24, 0.2, 0.24])

legend_ax.axis('off')

all_sizes = np.sort(all_sizes)
selected_sizes = all_sizes[[-1, 388, 296, 204, 112, 20]]

font_size = 18

legend_ax.scatter(0.3, 1, s=size_scale * selected_sizes[0], marker='*', color='black', edgecolor='k', clip_on=False)
legend_ax.text(0.305, 1, f'{selected_sizes[0]:.1f}', va='center', fontsize=font_size)

legend_ax.scatter(0.3, 0.9, s=size_scale * selected_sizes[1], marker='*', color='black', edgecolor='k')
legend_ax.text(0.305, 0.9, f'{selected_sizes[1]:.1f}', va='center', fontsize=font_size)

legend_ax.scatter(0.3, 0.8, s=size_scale * selected_sizes[2], marker='*', color='black', edgecolor='k')
legend_ax.text(0.305, 0.8, f'{selected_sizes[2]:.1f}', va='center', fontsize=font_size)

legend_ax.scatter(0.3, 0.7, s=size_scale * selected_sizes[3], marker='*', color='black', edgecolor='k')
legend_ax.text(0.305, 0.7, f'{selected_sizes[3]:.1f}', va='center', fontsize=font_size)

legend_ax.scatter(0.3, 0.6, s=size_scale * selected_sizes[4], marker='*', color='black', edgecolor='k')
legend_ax.text(0.305, 0.6, f'{selected_sizes[4]:.1f}', va='center', fontsize=font_size)

legend_ax.scatter(0.3, 0.5, s=size_scale * selected_sizes[5], marker='*', color='black', edgecolor='k')
legend_ax.text(0.305, 0.5, f'{selected_sizes[5]:.1f}', va='center', fontsize=font_size)

legend_ax.text(0.305, 0.314, 'Objective \n 5', fontsize=23, ha='center', fontname='Arial')

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.savefig('PF WFG3.svg', dpi=800)

plt.show()