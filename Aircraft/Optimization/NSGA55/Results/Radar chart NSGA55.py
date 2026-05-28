import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 55
names_all = ['x' + str(i + 1) for i in range(num_vars_all)]

bounds_all = np.zeros((num_vars_all, 2))
for i in range(num_vars_all):
    bounds_all[i, 1] = 1

whether_to_optimize = np.ones(num_vars_all)

num_vars = np.sum(whether_to_optimize == 1)
names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PS = []
for j in range(5):
    PS.append(np.load('PS-NSGA55-Aircraft-repetition-' + str(j) + '.npy'))

all_PS = np.array(np.concatenate(PS, axis=0))
all_PS = np.unique(all_PS, axis=0)

for j in range(all_PS.shape[1]):
    all_PS[:, j] = (all_PS[:, j] - bounds[j, 0]) / (bounds[j, 1] - bounds[j, 0])

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

PS_list = np.column_stack((all_PS, all_PS[:, 0]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(PS_list.shape[0]):
    line_color = mcolors.to_rgba('#70CDBE', alpha=0.3)
    marker_color = mcolors.to_rgba('#70CDBE', alpha=1.0)

    ax.plot(angles, PS_list[i, :], color=line_color, marker='o', markersize=2,
            markerfacecolor=marker_color, markeredgecolor=marker_color, linewidth=0.3, zorder=2)

radius_values = np.linspace(0, 1, num=5)
ax.set_yticks(radius_values)
ax.set_yticklabels([str(value) for value in radius_values],
                   fontdict={'fontsize': 12, 'family': 'Arial'}, zorder=1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(range(1, len(names) + 1), ha='center', fontsize=15, fontname='Arial')

plt.tight_layout()

plt.savefig('NSGA 55 PS Aircraft.svg', dpi=800,
            bbox_inches='tight')
plt.show()
