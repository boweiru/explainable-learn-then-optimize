import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from brokenaxes import brokenaxes
from matplotlib.ticker import MultipleLocator
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

rhd_NSGA60 = []
for i in range(10):
    rhd_NSGA60.append(np.load(os.path.join(root_dir, 'NSGA60\Results\RHD-NSGA60-Corridor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA60 = np.concatenate(rhd_NSGA60, axis=0)

rhd_NSGA9 = []
for i in range(10):
    rhd_NSGA9.append(np.load(os.path.join(root_dir, 'NSGA9\Results\RHD-NSGA9-Corridor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA9 = np.concatenate(rhd_NSGA9, axis=0)

rhd_MOBO60 = []
for i in range(10):
    rhd_MOBO60.append(np.load(os.path.join(root_dir, 'MOBO60\Results\RHD-MOBO60-Corridor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_MOBO60 = np.concatenate(rhd_MOBO60, axis=0)

rhd_MOBO9 = []
for i in range(10):
    rhd_MOBO9.append(np.load(os.path.join(root_dir, 'MOBO9\Results\RHD-MOBO9-Corridor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_TSO9 = np.concatenate(rhd_MOBO9, axis=0)

rhd_UniMOBO9 = []
for i in range(10):
    rhd_UniMOBO9.append(np.load(os.path.join(root_dir, 'XLO\Results\RHD-XLO-Corridor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_UniMOBO9 = np.concatenate(rhd_UniMOBO9, axis=0)

index_MOBO60 = np.arange(RHD_MOBO60.shape[1])
index_NB9 = np.arange(RHD_TSO9.shape[1])
index_NSGA60 = np.arange(RHD_NSGA60.shape[1]) * 10
index_NSGA9 = np.arange(RHD_NSGA9.shape[1]) * 10
index_UniMOBO9 = np.arange(RHD_UniMOBO9.shape[1]) - 20

std_RHD_MOBO60 = np.std(RHD_MOBO60, axis=0)
std_RHD_NB9 = np.std(RHD_TSO9, axis=0)
std_RHD_NSGA60 = np.std(RHD_NSGA60, axis=0)
std_RHD_NSGA9 = np.std(RHD_NSGA9, axis=0)
std_RHD_UniMOBO9 = np.std(RHD_UniMOBO9, axis=0)

fig = plt.figure(figsize=(12, 8))

bax = brokenaxes(
    xlims=((0, 0), (180, 500)),
    width_ratios=[0.00001, 1000],
    hspace=0.03,
    despine=True,
    d=0
)

line1 = bax.plot(index_NSGA60 + 200, np.mean(RHD_NSGA60, axis=0), linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                 label='NSGA-II (60)', linewidth=3.5, zorder=6)
line2 = bax.plot(index_MOBO60 + 200, np.mean(RHD_MOBO60, axis=0), linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                 label='MOBO (60)', linewidth=3.5, zorder=7)
line3 = bax.plot(index_NSGA9 + 200, np.mean(RHD_NSGA9, axis=0), linestyle='-', color=mcolors.to_rgba('#FFDD8E'),
                 label='NSGA-II (9)', linewidth=3.5, zorder=8)
line4 = bax.plot(index_NB9 + 200, np.mean(RHD_TSO9, axis=0), linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                 label='MOBO (9)', linewidth=3.5, zorder=9)
line5 = bax.plot(index_UniMOBO9 + 200, np.mean(RHD_UniMOBO9, axis=0), linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='XLO (9)', linewidth=3.5, zorder=10)

bax.fill_between(index_NSGA60 + 200, np.mean(RHD_NSGA60, axis=0) - std_RHD_NSGA60,
                 np.mean(RHD_NSGA60, axis=0) + std_RHD_NSGA60, color=mcolors.to_rgba('#70CDBE'), alpha=0.2,
                 zorder=2)
bax.fill_between(index_MOBO60 + 200, np.mean(RHD_MOBO60, axis=0) - std_RHD_MOBO60,
                 np.mean(RHD_MOBO60, axis=0) + std_RHD_MOBO60, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2,
                 zorder=3)
bax.fill_between(index_NSGA9 + 200, np.mean(RHD_NSGA9, axis=0) - std_RHD_NSGA9,
                 np.mean(RHD_NSGA9, axis=0) + std_RHD_NSGA9, color=mcolors.to_rgba('#FFDD8E'), alpha=0.2,
                 zorder=1)
bax.fill_between(index_NB9 + 200, np.mean(RHD_TSO9, axis=0) - std_RHD_NB9,
                 np.mean(RHD_TSO9, axis=0) + std_RHD_NB9, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2,
                 zorder=4)
bax.fill_between(index_UniMOBO9 + 200, np.mean(RHD_UniMOBO9, axis=0) - std_RHD_UniMOBO9,
                 np.mean(RHD_UniMOBO9, axis=0) + std_RHD_UniMOBO9, color=mcolors.to_rgba('#AC99D2'), alpha=0.2,
                 zorder=5)

bax.set_xlabel('Number of sample point evaluations', fontsize=28, labelpad=35, fontname='Arial')
bax.set_ylabel('Relative hypervolume ratio', fontsize=28, labelpad=72, fontname='Arial')
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

plt.subplots_adjust(
    right=0.975,
)

bax.axs[0].yaxis.set_major_locator(MultipleLocator(0.005))
bax.axs[1].yaxis.set_major_locator(MultipleLocator(0.005))


plt.savefig('RHR Corridor.svg', dpi=800)
plt.show()


