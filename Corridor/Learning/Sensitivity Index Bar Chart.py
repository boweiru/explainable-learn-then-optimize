import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams

# 数据
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
names_all = np.array(names_all)

PT = np.load('PT.npy')

numbers = 20
sorted_idx = np.argsort(-PT)[:numbers]
names_all = names_all[sorted_idx][:numbers]
PT = PT[sorted_idx][:numbers]

cum = np.load('cum_PT_descending.npy')[:numbers]

bar_width = 0.7
x = np.arange(numbers)

fig, ax1 = plt.subplots(figsize=(16, 10))


# 柱状图（左Y轴）
bars = ax1.bar(x, PT, color=mcolors.to_rgba('#B7E8DF'), edgecolor=mcolors.to_rgba('#4C9F89'),
               label=r'$\mathit{\hat{P}^{\mathrm{total}}}$', linewidth=4,
               width=bar_width,
               align='center', zorder=2
               )


ax1.set_xlabel('Factor index', fontsize=35, fontname='Arial', labelpad=10)
ax1.set_ylabel('Generalized sensitivity indices', fontsize=35, fontname='Arial', labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(sorted_idx + 1, ha='center', fontsize=30, fontname='Arial')
ax1.tick_params(axis='both', labelsize=30, length=8, width=2.5)
plt.setp(ax1.get_yticklabels(), fontname='Arial')

ax2 = ax1.twinx()

ax2.plot(x, cum, linestyle='-', marker='o', color=mcolors.to_rgba('#7A66B9'),
         label=r'$\hat{P}^{\mathrm{total}}_{i_1 \dots i_s}$', markersize=15,
         markerfacecolor='white', markeredgewidth=3,
         markeredgecolor=mcolors.to_rgba('#7A66B9'), linewidth=4, zorder=2)

ax2.set_ylabel('Cumulative variance contribution', fontsize=35, fontname='Arial',
               labelpad=10, rotation=270, va='bottom')
ax2.tick_params(axis='both', labelsize=30, length=8, width=2.5)
plt.setp(ax2.get_yticklabels(), fontname='Arial')

lines, labels = [], []
for ax in [ax1, ax2]:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_linewidth(2.5)
ax1.spines['right'].set_linewidth(2.5)
ax1.spines['bottom'].set_linewidth(2.5)
ax2.yaxis.label.set_color('#7A66B9')


legend = ax1.legend(
    lines, labels,
    loc='lower right',
    frameon=True,
    edgecolor='black',
    fontsize=35,
    handlelength=3,
    handletextpad=0.8,
    borderaxespad=1.2,
    labelspacing=0.5,
    markerscale=1.5
)
legend.get_frame().set_linewidth(2.5)

plt.tight_layout()
plt.savefig('Sensitivity Corridor.svg', dpi=800,
            bbox_inches='tight')
plt.show()
