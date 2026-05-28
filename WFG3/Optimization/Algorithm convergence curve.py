import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from brokenaxes import brokenaxes
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

rhd_EIM100 = []
for i in [0, 1, 3, 4, 5, 6, 7, 9]:
    rhd_EIM100.append(np.load(os.path.join(root_dir, 'MOBO100\Results\RHD-MOBO100-WFG3-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_EIM100 = np.concatenate(rhd_EIM100, axis=0)

rhd_NSGA100 = []
for i in range(10):
    rhd_NSGA100.append(np.load(os.path.join(root_dir, 'NSGA100\Results\RHD-NSGA100-WFG3-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA100 = np.concatenate(rhd_NSGA100, axis=0)

rhd_NSGA4 = []
for i in range(10):
    rhd_NSGA4.append(np.load(os.path.join(root_dir, 'NSGA4\Results\RHD-NSGA4-WFG3-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA4 = np.concatenate(rhd_NSGA4, axis=0)

rhd_TSO4 = []
for i in range(10):
    rhd_TSO4.append(np.load(os.path.join(root_dir, 'MOBO4\Results\RHD-MOBO4-WFG3-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NB4 = np.concatenate(rhd_TSO4, axis=0)

rhd_XLO4 = []
for i in range(10):
    rhd_XLO4.append(np.load(os.path.join(root_dir, 'XLO\Results\RHD-XLO-WFG3-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_XLO4 = np.concatenate(rhd_XLO4, axis=0)

index_EIM100 = np.arange(RHD_EIM100.shape[1])
index_NB4 = np.arange(RHD_NB4.shape[1])
index_NSGA100 = np.arange(RHD_NSGA100.shape[1]) * 10
index_NSGA4 = np.arange(RHD_NSGA4.shape[1]) * 10
index_XLO4 = np.arange(RHD_XLO4.shape[1]) - 10

std_RHD_EIM100 = np.std(RHD_EIM100, axis=0)
std_RHD_NB4 = np.std(RHD_NB4, axis=0)
std_RHD_NSGA100 = np.std(RHD_NSGA100, axis=0)
std_RHD_NSGA4 = np.std(RHD_NSGA4, axis=0)
std_RHD_XLO4 = np.std(RHD_XLO4, axis=0)

fig = plt.figure(figsize=(12, 8))
bax = brokenaxes(
    xlims=((0, 0), (400, 1000)),
    width_ratios=[0.00001, 1000],
    hspace=0.03,
    despine=True,
    d=0
)

line1 = bax.plot(index_NSGA100 + 410, np.mean(RHD_NSGA100, axis=0), linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                 label='NSGA-II (100)', linewidth=3.5, zorder=6)
line2 = bax.plot(index_EIM100 + 410, np.mean(RHD_EIM100, axis=0), linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                 label='MOBO (100)', linewidth=3.5, zorder=7)
line3 = bax.plot(index_NSGA4 + 410, np.mean(RHD_NSGA4, axis=0), linestyle='-', color=mcolors.to_rgba('#FFDD8E'),
                 label='NSGA-II (4)', linewidth=3.5, zorder=8)
line4 = bax.plot(index_NB4 + 410, np.mean(RHD_NB4, axis=0), linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                 label='MOBO (4)', linewidth=3.5, zorder=9)
line5 = bax.plot(index_XLO4 + 410, np.mean(RHD_XLO4, axis=0), linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='XLO (4)', linewidth=3.5, zorder=10)

bax.fill_between(index_NSGA100 + 410, np.mean(RHD_NSGA100, axis=0) - std_RHD_NSGA100,
                 np.mean(RHD_NSGA100, axis=0) + std_RHD_NSGA100, color=mcolors.to_rgba('#70CDBE'), alpha=0.2,
                 zorder=2)
bax.fill_between(index_EIM100 + 410, np.mean(RHD_EIM100, axis=0) - std_RHD_EIM100,
                 np.mean(RHD_EIM100, axis=0) + std_RHD_EIM100, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2,
                 zorder=3)
bax.fill_between(index_NSGA4 + 410, np.mean(RHD_NSGA4, axis=0) - std_RHD_NSGA4,
                 np.mean(RHD_NSGA4, axis=0) + std_RHD_NSGA4, color=mcolors.to_rgba('#FFDD8E'), alpha=0.2,
                 zorder=1)
bax.fill_between(index_NB4 + 410, np.mean(RHD_NB4, axis=0) - std_RHD_NB4,
                 np.mean(RHD_NB4, axis=0) + std_RHD_NB4, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2,
                 zorder=4)
bax.fill_between(index_XLO4 + 410, np.mean(RHD_XLO4, axis=0) - std_RHD_XLO4,
                 np.mean(RHD_XLO4, axis=0) + std_RHD_XLO4, color=mcolors.to_rgba('#AC99D2'), alpha=0.2,
                 zorder=5)

bax.set_xlabel('Number of sample point evaluations', fontsize=28, labelpad=35, fontname='Arial')
bax.set_ylabel('Relative hypervolume ratio', fontsize=28, labelpad=65, fontname='Arial')
bax.tick_params(axis='both', labelsize=22, length=8, width=2.5)

for ax in bax.axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)

font_prop = FontProperties(
    family='Arial',
    size=28,
    weight='normal'
)

legend = bax.legend(
    bbox_to_anchor=(1, 0.85),
    frameon=True,
    edgecolor='black',
    prop=font_prop,
    handlelength=3,
    handletextpad=0.8,
    borderaxespad=0.3,
    labelspacing=0.3,
    markerscale=1.3
)
legend.get_frame().set_linewidth(2.5)

plt.subplots_adjust(
    right=0.96
)

plt.savefig('RHR WFG3.svg', dpi=800)
plt.show()