import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from brokenaxes import brokenaxes
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

rhd_NSGA55 = []
for i in range(5):
    rhd_NSGA55.append(np.load(os.path.join(root_dir, 'NSGA55\Results\RHD-NSGA55-Aircraft-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA55 = np.concatenate(rhd_NSGA55, axis=0)

rhd_NSGA5 = []
for i in range(5):
    rhd_NSGA5.append(np.load(os.path.join(root_dir, 'NSGA5\Results\RHD-NSGA5-Aircraft-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA5 = np.concatenate(rhd_NSGA5, axis=0)

rhd_MOBO55 = []
for i in range(5):
    rhd_MOBO55.append(np.load(os.path.join(root_dir,
        'MOBO55\Results\RHD-MOBO55-Aircraft-repetition-' + str(
            i) + '.npy')).reshape(1, -1))
RHD_MOBO55 = np.concatenate(rhd_MOBO55, axis=0)

rhd_MOBO5 = []
for i in range(5):
    rhd_MOBO5.append(np.load(os.path.join(root_dir, 'MOBO5\Results\RHD-MOBO5-Aircraft-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_TSO5 = np.concatenate(rhd_MOBO5, axis=0)

rhd_UniMOBO5 = []
for i in range(5):
    rhd_UniMOBO5.append(np.load(os.path.join(root_dir, 'XLO\Results\RHD-XLO-Aircraft-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_UniMOBO5 = np.concatenate(rhd_UniMOBO5, axis=0)

index_MOBO55 = np.arange(RHD_MOBO55.shape[1])
index_NB5 = np.arange(RHD_TSO5.shape[1])
index_NSGA55 = np.arange(RHD_NSGA55.shape[1]) * 10
index_NSGA5 = np.arange(RHD_NSGA5.shape[1]) * 10
index_UniMOBO5 = np.arange(RHD_UniMOBO5.shape[1]) - 10

std_RHD_MOBO55 = np.std(RHD_MOBO55, axis=0)
std_RHD_NB5 = np.std(RHD_TSO5, axis=0)
std_RHD_NSGA55 = np.std(RHD_NSGA55, axis=0)
std_RHD_NSGA5 = np.std(RHD_NSGA5, axis=0)
std_RHD_UniMOBO5 = np.std(RHD_UniMOBO5, axis=0)

fig = plt.figure(figsize=(12, 8))
bax = brokenaxes(
    xlims=((0, 0), (180, 500)),
    width_ratios=[0.00001, 1000],
    hspace=0.03,
    despine=True,
    d=0
)

line1 = bax.plot(index_NSGA55 + 190, np.mean(RHD_NSGA55, axis=0), linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                 label='NSGA-II (55)', linewidth=3.5, zorder=6)
line2 = bax.plot(index_MOBO55 + 190, np.mean(RHD_MOBO55, axis=0), linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                 label='MOBO (55)', linewidth=3.5, zorder=7)
line3 = bax.plot(index_NSGA5 + 190, np.mean(RHD_NSGA5, axis=0), linestyle='-', color=mcolors.to_rgba('#FFDD8E'),
                 label='NSGA-II (5)', linewidth=3.5, zorder=8)
line4 = bax.plot(index_NB5 + 190, np.mean(RHD_TSO5, axis=0), linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                 label='MOBO (5)', linewidth=3.5, zorder=9)
line5 = bax.plot(index_UniMOBO5 + 190, np.mean(RHD_UniMOBO5, axis=0), linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='XLO (5)', linewidth=3.5, zorder=10)

bax.fill_between(index_NSGA55 + 190, np.mean(RHD_NSGA55, axis=0) - std_RHD_NSGA55,
                 np.mean(RHD_NSGA55, axis=0) + std_RHD_NSGA55, color=mcolors.to_rgba('#70CDBE'), alpha=0.2,
                 zorder=2)
bax.fill_between(index_MOBO55 + 190, np.mean(RHD_MOBO55, axis=0) - std_RHD_MOBO55,
                 np.mean(RHD_MOBO55, axis=0) + std_RHD_MOBO55, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2,
                 zorder=3)
bax.fill_between(index_NSGA5 + 190, np.mean(RHD_NSGA5, axis=0) - std_RHD_NSGA5,
                 np.mean(RHD_NSGA5, axis=0) + std_RHD_NSGA5, color=mcolors.to_rgba('#FFDD8E'), alpha=0.2,
                 zorder=1)
bax.fill_between(index_NB5 + 190, np.mean(RHD_TSO5, axis=0) - std_RHD_NB5,
                 np.mean(RHD_TSO5, axis=0) + std_RHD_NB5, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2,
                 zorder=4)
bax.fill_between(index_UniMOBO5 + 190, np.mean(RHD_UniMOBO5, axis=0) - std_RHD_UniMOBO5,
                 np.mean(RHD_UniMOBO5, axis=0) + std_RHD_UniMOBO5, color=mcolors.to_rgba('#AC99D2'), alpha=0.2,
                 zorder=5)

bax.set_xlabel('Number of sample point evaluations', fontsize=28, labelpad=35, fontname='Arial')
bax.set_ylabel('Relative hypervolume ratio', fontsize=28, labelpad=80, fontname='Arial')
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

plt.savefig('RHR Aircraft.svg', dpi=800)
plt.show()


