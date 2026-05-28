import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 60

names_all = []
names_all.append('psg_length')
names_all.append('psg_speedFactor')
names_all.append('psg_speedDev')
names_all.append('psg_minGap')
names_all.append('psg_accel')
names_all.append('psg_decel')
names_all.append('psg_tau')
names_all.append('psg_delta')
names_all.append('psg_emergencyDecel')
names_all.append('psg_lcStrategic')
names_all.append('psg_lcCooperative')
names_all.append('psg_lcSpeedGain')
names_all.append('psg_lcKeepRight')
names_all.append('psg_lcOvertakeRight')
names_all.append('psg_lcOpposite')
names_all.append('psg_lcLookaheadLeft')
names_all.append('psg_lcSpeedGainRight')
names_all.append('psg_lcSpeedGainLookahead')
names_all.append('psg_lcOvertakeDeltaSpeedFactor')
names_all.append('psg_lcKeepRightAcceptanceTime')
names_all.append('psg_lcCooperativeSpeed')
names_all.append('psg_lcAssertive')
names_all.append('psg_jmCrossingGap')
names_all.append('psg_jmIgnoreFoeProb')
names_all.append('psg_jmIgnoreFoeSpeed')
names_all.append('psg_jmIgnoreJunctionFoeProb')
names_all.append('psg_jmSigmaMinor')
names_all.append('psg_jmStoplineGap')
names_all.append('psg_jmTimegapMinor')
names_all.append('psg_impatience')
names_all.append('trk_length')
names_all.append('trk_speedFactor')
names_all.append('trk_speedDev')
names_all.append('trk_minGap')
names_all.append('trk_accel')
names_all.append('trk_decel')
names_all.append('trk_tau')
names_all.append('trk_delta')
names_all.append('trk_emergencyDecel')
names_all.append('trk_lcStrategic')
names_all.append('trk_lcCooperative')
names_all.append('trk_lcSpeedGain')
names_all.append('trk_lcKeepRight')
names_all.append('trk_lcOvertakeRight')
names_all.append('trk_lcOpposite')
names_all.append('trk_lcLookaheadLeft')
names_all.append('trk_lcSpeedGainRight')
names_all.append('trk_lcSpeedGainLookahead')
names_all.append('trk_lcOvertakeDeltaSpeedFactor')
names_all.append('trk_lcKeepRightAcceptanceTime')
names_all.append('trk_lcCooperativeSpeed')
names_all.append('trk_lcAssertive')
names_all.append('trk_jmCrossingGap')
names_all.append('trk_jmIgnoreFoeProb')
names_all.append('trk_jmIgnoreFoeSpeed')
names_all.append('trk_jmIgnoreJunctionFoeProb')
names_all.append('trk_jmSigmaMinor')
names_all.append('trk_jmStoplineGap')
names_all.append('trk_jmTimegapMinor')
names_all.append('trk_impatience')

bounds_all = [[3, 7],
              [0.2, 2],
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
              [0, 10],
              [7, 12],
              [0.2, 2],
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
whether_to_optimize[3] = 1
whether_to_optimize[7] = 1
whether_to_optimize[1] = 1
whether_to_optimize[21] = 1
whether_to_optimize[4] = 1
whether_to_optimize[6] = 1
whether_to_optimize[2] = 1
whether_to_optimize[25] = 1
whether_to_optimize[0] = 1

num_vars = np.sum(whether_to_optimize == 1)
names = [r'$x_{1}$', r'$x_{2}$', r'$x_{3}$', r'$x_{4}$', r'$x_{5}$', r'$x_{7}$', r'$x_{8}$', r'$x_{22}$', r'$x_{26}$']
bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

PS = []
for j in range(10):
    PS.append(np.load('PS-NSGA9-Corridor-repetition-' + str(j) + '.npy'))

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

plt.savefig('NSGA 9 PS Corridor.svg', dpi=800,
            bbox_inches='tight')
plt.show()
