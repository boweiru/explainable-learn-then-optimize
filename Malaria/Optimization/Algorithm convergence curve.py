import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from brokenaxes import brokenaxes
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

rhd_MOBO23 = []
for i in range(10):
    rhd_MOBO23.append(np.load(os.path.join(root_dir, 'MOBO23\Results\RHD-MOBO23-Malaria-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_MOBO23 = np.concatenate(rhd_MOBO23, axis=0)

rhd_MOBO5 = []
for i in range(10):
    rhd_MOBO5.append(np.load(os.path.join(root_dir, 'MOBO5\Results\RHD-MOBO5-Malaria-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_MOBO5 = np.concatenate(rhd_MOBO5, axis=0)

rhd_NSGA23 = []
for i in range(10):
    rhd_NSGA23.append(np.load(os.path.join(root_dir, 'NSGA23\Results\RHD-NSGA23-Malaria-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA23 = np.concatenate(rhd_NSGA23, axis=0)

rhd_NSGA5 = []
for i in range(10):
    rhd_NSGA5.append(np.load(os.path.join(root_dir, 'NSGA5\Results\RHD-NSGA5-Malaria-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA5 = np.concatenate(rhd_NSGA5, axis=0)

rhd_XLO5 = []
for i in range(10):
    rhd_XLO5.append(np.load(os.path.join(root_dir, 'XLO\Results\RHD-XLO-Malaria-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_XLO5 = np.concatenate(rhd_XLO5, axis=0)

index_MOBO23 = np.arange(RHD_MOBO23.shape[1])
index_MOBO5 = np.arange(RHD_MOBO5.shape[1])
index_NSGA23 = np.arange(RHD_NSGA23.shape[1]) * 10
index_NSGA5 = np.arange(RHD_NSGA5.shape[1]) * 10
index_XLO5 = np.arange(RHD_XLO5.shape[1]) - 10

std_RHD_MOBO23 = np.std(RHD_MOBO23, axis=0)
std_RHD_MOBO5 = np.std(RHD_MOBO5, axis=0)
std_RHD_NSGA23 = np.std(RHD_NSGA23, axis=0)
std_RHD_NSGA5 = np.std(RHD_NSGA5, axis=0)
std_RHD_XLO5 = np.std(RHD_XLO5, axis=0)

fig = plt.figure(figsize=(12, 8))
bax = brokenaxes(
    xlims=((0, 0), (40, 200)),
    width_ratios=[0.00001, 1000],
    hspace=0.03,
    despine=True,
    d=0
)

line1 = bax.plot(index_NSGA23 + 50, np.mean(RHD_NSGA23, axis=0), linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                 label='NSGA-II (23)', linewidth=3.5, zorder=6)
line2 = bax.plot(index_MOBO23 + 50, np.mean(RHD_MOBO23, axis=0), linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                 label='MOBO (23)', linewidth=3.5, zorder=7)
line3 = bax.plot(index_NSGA5 + 50, np.mean(RHD_NSGA5, axis=0), linestyle='-', color=mcolors.to_rgba('#FFDD8E'),
                 label='NSGA-II (5)', linewidth=3.5, zorder=8)
line4 = bax.plot(index_MOBO5 + 50, np.mean(RHD_MOBO5, axis=0), linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                 label='MOBO (5)', linewidth=3.5, zorder=9)
line5 = bax.plot(index_XLO5 + 50, np.mean(RHD_XLO5, axis=0), linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='XLO (5)', linewidth=3.5, zorder=10)

bax.fill_between(index_NSGA23 + 50, np.mean(RHD_NSGA23, axis=0) - std_RHD_NSGA23,
                 np.mean(RHD_NSGA23, axis=0) + std_RHD_NSGA23, color=mcolors.to_rgba('#70CDBE'), alpha=0.2,
                 zorder=2)
bax.fill_between(index_MOBO23 + 50, np.mean(RHD_MOBO23, axis=0) - std_RHD_MOBO23,
                 np.mean(RHD_MOBO23, axis=0) + std_RHD_MOBO23, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2,
                 zorder=3)
bax.fill_between(index_NSGA5 + 50, np.mean(RHD_NSGA5, axis=0) - std_RHD_NSGA5,
                 np.mean(RHD_NSGA5, axis=0) + std_RHD_NSGA5, color=mcolors.to_rgba('#FFDD8E'), alpha=0.2,
                 zorder=1)
bax.fill_between(index_MOBO5 + 50, np.mean(RHD_MOBO5, axis=0) - std_RHD_MOBO5,
                 np.mean(RHD_MOBO5, axis=0) + std_RHD_MOBO5, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2,
                 zorder=4)
bax.fill_between(index_XLO5 + 50, np.mean(RHD_XLO5, axis=0) - std_RHD_XLO5,
                 np.mean(RHD_XLO5, axis=0) + std_RHD_XLO5, color=mcolors.to_rgba('#AC99D2'), alpha=0.2,
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
    loc='upper right',
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
legend.get_frame().set_facecolor('none')

plt.subplots_adjust(
    right=0.97,
)

plt.savefig('RHR Malaria.svg', dpi=800)
plt.show()