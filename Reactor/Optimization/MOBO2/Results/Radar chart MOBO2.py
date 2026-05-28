import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 50
names_all = ['x' + str(i + 1) for i in range(num_vars_all)]

bounds_all = np.zeros((num_vars_all, 2))
for i in range(num_vars_all):
    bounds_all[i, 1] = 1

whether_to_optimize = np.zeros(num_vars_all)
whether_to_optimize[0] = 1
whether_to_optimize[2] = 1

num_vars = np.sum(whether_to_optimize == 1)
names = [r'$x_{1}$', r'$x_{3}$']
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PS = []
for j in range(5):
    PS.append(np.load('PS-MOBO2-Reactor-repetition-' + str(j) + '.npy'))

all_PS = np.array(np.concatenate(PS, axis=0))
all_PS = np.unique(all_PS, axis=0)

for j in range(all_PS.shape[1]):
    all_PS[:, j] = (all_PS[:, j] - bounds[j, 0]) / (bounds[j, 1] - bounds[j, 0])

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

PS_list = np.column_stack((all_PS, all_PS[:, 0]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(PS_list.shape[0]):
    line_color = mcolors.to_rgba('#8FB4DC', alpha=0.5)
    marker_color = mcolors.to_rgba('#8FB4DC', alpha=1.0)

    ax.plot(angles, PS_list[i, :], color=line_color, marker='o', markersize=2,
            markerfacecolor=marker_color, markeredgecolor=marker_color, linewidth=0.5, zorder=2)

radius_values = np.linspace(0, 1, num=5)
ax.set_yticks(radius_values)
ax.set_yticklabels([str(value) for value in radius_values],
                   fontdict={'fontsize': 12, 'family': 'Arial'}, zorder=1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(names, ha='center', fontsize=15, fontname='Arial')

plt.tight_layout()

plt.savefig('MOBO 2 PS Reactor.svg', dpi=800,
            bbox_inches='tight')
plt.show()
