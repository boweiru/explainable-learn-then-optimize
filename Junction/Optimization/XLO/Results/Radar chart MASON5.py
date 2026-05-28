import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 29

names_all = []
names_all.append('speedFactor')
names_all.append('speedDev')
names_all.append('minGap')
names_all.append('accel')
names_all.append('decel')
names_all.append('tau')
names_all.append('delta')
names_all.append('emergencyDecel')
names_all.append('lcStrategic')
names_all.append('lcCooperative')
names_all.append('lcSpeedGain')
names_all.append('lcKeepRight')
names_all.append('lcOvertakeRight')
names_all.append('lcOpposite')
names_all.append('lcLookaheadLeft')
names_all.append('lcSpeedGainRight')
names_all.append('lcSpeedGainLookahead')
names_all.append('lcOvertakeDeltaSpeedFactor')
names_all.append('lcKeepRightAcceptanceTime')
names_all.append('lcCooperativeSpeed')
names_all.append('lcAssertive')
names_all.append('jmCrossingGap')
names_all.append('jmIgnoreFoeProb')
names_all.append('jmIgnoreFoeSpeed')
names_all.append('jmIgnoreJunctionFoeProb')
names_all.append('jmSigmaMinor')
names_all.append('jmStoplineGap')
names_all.append('jmTimegapMinor')
names_all.append('impatience')

bounds_all = [[0.2, 2],
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
              [0, 10]]
bounds_all = np.array(bounds_all)

whether_to_optimize = np.zeros(num_vars_all)
whether_to_optimize[0] = 1
whether_to_optimize[2] = 1
whether_to_optimize[3] = 1
whether_to_optimize[5] = 1
whether_to_optimize[6] = 1

num_vars = np.sum(whether_to_optimize == 1)
names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PS = []
for j in range(10):
    PS.append(np.load('PS-XLO-Junction-repetition-' + str(j) + '.npy'))

all_PS = np.array(np.concatenate(PS, axis=0))
all_PS = np.unique(all_PS, axis=0)
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
            markerfacecolor=marker_color, markeredgecolor=marker_color, linewidth=0.3, zorder=2)

radius_values = np.linspace(0, 1, num=5)
ax.set_yticks(radius_values)
ax.set_yticklabels([str(value) for value in radius_values],
                   fontdict={'fontsize': 12, 'family': 'Arial'}, zorder=1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(names, ha='center', fontsize=15, fontname='Arial')

plt.tight_layout()

plt.savefig('XLO 5 PS Junction.svg', dpi=800,
            bbox_inches='tight')
plt.show()
