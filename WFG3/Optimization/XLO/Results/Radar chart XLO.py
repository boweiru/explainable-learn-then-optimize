import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from deap import base, creator, tools


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

    elite_indices = []
    for ef in elite_F:
        index = np.where(np.all(F == ef, axis=1))[0][0]
        elite_indices.append(index)

    return elite_F, np.array(elite_indices)


num_vars_all = 100
k = 4

names_all = [r'$x_{' + str(i + 1) + '}$' for i in range(num_vars_all)]
names_all = np.array(names_all)

bounds_all = np.zeros((num_vars_all, 2))
for i in range(num_vars_all):
    bounds_all[i, 1] = 2 * (i + 1)

default_values = np.array([float(2 * (i + 1) * 0.35) for i in range(num_vars_all)])

whether_to_optimize = np.zeros(num_vars_all)
for i in range(k):
    whether_to_optimize[i] = 1

num_vars = np.sum(whether_to_optimize == 1)
names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PF_TSO4 = []
for i in range(10):
    PF_TSO4.append(np.load('PF-XLO-WFG3-repetition-' + str(i) + '.npy'))
all_PF_TSO4 = np.concatenate(PF_TSO4, axis=0)
all_PF_TSO4 = np.unique(all_PF_TSO4, axis=0)
all_PF_TSO4, elite_indices = nsga2_elite_selection(all_PF_TSO4, 100)

PS = []
for j in range(10):
    PS.append(np.load('PS-XLO-WFG3-repetition-' + str(j) + '.npy'))

all_PS = np.array(np.concatenate(PS, axis=0))
all_PS = np.unique(all_PS, axis=0)
all_PS = all_PS[elite_indices]
all_PS = all_PS[:, whether_to_optimize == 1]

for j in range(all_PS.shape[1]):
    all_PS[:, j] = (all_PS[:, j] - bounds[j, 0]) / (bounds[j, 1] - bounds[j, 0])

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

PS_list = np.column_stack((all_PS, all_PS[:, 0]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(PS_list.shape[0]):
    line_color = mcolors.to_rgba('#AC99D2', alpha=0.5)
    marker_color = mcolors.to_rgba('#AC99D2', alpha=1.0)

    ax.plot(angles, PS_list[i, :], color=line_color, marker='o', markersize=2,
            markerfacecolor=marker_color, markeredgecolor=marker_color, linewidth=0.5, zorder=2)

radius_values = np.linspace(0, 1, num=5)
ax.set_yticks(radius_values)
ax.set_yticklabels([str(value) for value in radius_values],
                   fontdict={'fontsize': 12, 'family': 'Arial'}, zorder=1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(names, ha='center', fontsize=15, fontname='Arial')

plt.tight_layout()

plt.savefig('XLO 4 PS WFG3.svg', dpi=800,
            bbox_inches='tight')
plt.show()
