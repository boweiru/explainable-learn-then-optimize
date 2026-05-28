import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 23

names_all = [
        "Simm", "Xp", "Yp", "sigma2_i", "Xy_star", "Xh_star",
        "ln_1_minus_alpha_m", "alpha_m_decay", "sigma2_0", "Xv_star",
        "Y2_star", "alpha", "v1", "log_phi1", "Q_D", "Q_n",
        "v0", "YB1_star", "F0", "log2_over_omega",
        "Y1_star", "Y0_star", "alpha_F_star",
    ]
bounds_all = [
        [0, 6],  # Simm
        [0, 10000],  # Xp
        [0, 14],  # Yp
        [1, 75],  # sigma2_i
        [0, 1000],  # Xy_star
        [0, 200],  # Xh_star
        [0, 5],  # ln_1_minus_alpha_m
        [0, 11],  # alpha_m_decay
        [0, 26],  # sigma2_0
        [0, 66],  # Xv_star
        [0, 20000],  # Y2_star
        [10000, 200000],  # alpha
        [0, 2],  # v1
        [0, 9],  # log_phi1
        [0, 1],  # Q_D
        [3, 651],  # Q_n
        [1, 16],  # v0
        [20000, 1600000],  # YB1_star
        [0, 1],  # F0
        [0, 4],  # log2_over_omega
        [0, 10],  # Y1_star
        [0, 600],  # Y0_star
        [0, 3],  # alpha_F_star
    ]
bounds_all = np.array(bounds_all)

whether_to_optimize = np.zeros(num_vars_all)
whether_to_optimize[3] = 1
whether_to_optimize[8] = 1
whether_to_optimize[11] = 1
whether_to_optimize[14] = 1
whether_to_optimize[19] = 1

num_vars = np.sum(whether_to_optimize == 1)
names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PS = []
for j in range(10):
    PS.append(np.load('PS-NSGA5-malaria-repetition-' + str(j) + '.npy'))

all_PS = np.array(np.concatenate(PS, axis=0))
all_PS = np.unique(all_PS, axis=0)

for j in range(all_PS.shape[1]):
    all_PS[:, j] = (all_PS[:, j] - bounds[j, 0]) / (bounds[j, 1] - bounds[j, 0])

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

PS_list = np.column_stack((all_PS, all_PS[:, 0]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i in range(PS_list.shape[0]):
    line_color = mcolors.to_rgba('#FFDD8E', alpha=0.5)
    marker_color = mcolors.to_rgba('#FFDD8E', alpha=1.0)

    ax.plot(angles, PS_list[i, :], color=line_color, marker='o', markersize=2,
            markerfacecolor=marker_color, markeredgecolor=marker_color, linewidth=0.5, zorder=2)

radius_values = np.linspace(0, 1, num=5)
ax.set_yticks(radius_values)
ax.set_yticklabels([str(value) for value in radius_values],
                   fontdict={'fontsize': 12, 'family': 'Arial'}, zorder=1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(names, ha='center', fontsize=15, fontname='Arial')

plt.tight_layout()


plt.savefig('NSGA 5 PS Malaria.svg', dpi=800,
            bbox_inches='tight')
plt.show()
