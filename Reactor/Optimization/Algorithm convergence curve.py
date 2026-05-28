import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from brokenaxes import brokenaxes
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

rhd_NSGA50 = []
for i in range(5):
    rhd_NSGA50.append(np.load(os.path.join(root_dir, 'NSGA50\Results\RHD-NSGA50-Reactor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA50 = np.concatenate(rhd_NSGA50, axis=0)

rhd_NSGA2 = []
for i in range(5):
    rhd_NSGA2.append(np.load(os.path.join(root_dir, 'NSGA2\Results\RHD-NSGA2-Reactor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_NSGA2 = np.concatenate(rhd_NSGA2, axis=0)

rhd_MOBO50 = []
for i in range(5):
    rhd_MOBO50.append(np.load(os.path.join(root_dir,
        'MOBO50\Results\RHD-MOBO50-Reactor-repetition-' + str(
            i) + '.npy')).reshape(1, -1))
RHD_MOBO50 = np.concatenate(rhd_MOBO50, axis=0)

rhd_MOBO2 = []
for i in range(5):
    rhd_MOBO2.append(np.load(os.path.join(root_dir, 'MOBO2\Results\RHD-MOBO2-Reactor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_TSO2 = np.concatenate(rhd_MOBO2, axis=0)

rhd_XLO2 = []
for i in range(5):
    rhd_XLO2.append(np.load(os.path.join(root_dir, 'XLO\Results\RHD-XLO-Reactor-repetition-' + str(i) + '.npy')).reshape(1, -1))
RHD_XLO2 = np.concatenate(rhd_XLO2, axis=0)

index_MOBO50 = np.arange(RHD_MOBO50.shape[1])
index_NB2 = np.arange(RHD_TSO2.shape[1])
index_NSGA50 = np.arange(RHD_NSGA50.shape[1]) * 10
index_NSGA2 = np.arange(RHD_NSGA2.shape[1]) * 10
index_XLO2 = np.arange(RHD_XLO2.shape[1]) - 10

std_RHD_MOBO50 = np.std(RHD_MOBO50, axis=0)
std_RHD_NB2 = np.std(RHD_TSO2, axis=0)
std_RHD_NSGA50 = np.std(RHD_NSGA50, axis=0)
std_RHD_NSGA2 = np.std(RHD_NSGA2, axis=0)
std_RHD_XLO2 = np.std(RHD_XLO2, axis=0)

fig = plt.figure(figsize=(12, 8))
bax = brokenaxes(
    xlims=((0, 0), (150, 500)),
    width_ratios=[0.00001, 1000],
    hspace=0.03,
    despine=True,
    d=0
)

line1 = bax.plot(index_NSGA50 + 160, np.mean(RHD_NSGA50, axis=0), linestyle='-', color=mcolors.to_rgba('#70CDBE'),
                 label='NSGA-II (50)', linewidth=3.5, zorder=6)
line2 = bax.plot(index_MOBO50 + 160, np.mean(RHD_MOBO50, axis=0), linestyle='-', color=mcolors.to_rgba('#F9B3AD'),
                 label='MOBO (50)', linewidth=3.5, zorder=7)
line3 = bax.plot(index_NSGA2 + 160, np.mean(RHD_NSGA2, axis=0), linestyle='-', color=mcolors.to_rgba('#FFDD8E'),
                 label='NSGA-II (2)', linewidth=3.5, zorder=8)
line4 = bax.plot(index_NB2 + 160, np.mean(RHD_TSO2, axis=0), linestyle='-', color=mcolors.to_rgba('#8FB4DC'),
                 label='MOBO (2)', linewidth=3.5, zorder=9)
line5 = bax.plot(index_XLO2 + 160, np.mean(RHD_XLO2, axis=0), linestyle='-', color=mcolors.to_rgba('#AC99D2'),
                 label='XLO (2)', linewidth=3.5, zorder=10)

bax.fill_between(index_NSGA50 + 160, np.mean(RHD_NSGA50, axis=0) - std_RHD_NSGA50,
                 np.mean(RHD_NSGA50, axis=0) + std_RHD_NSGA50, color=mcolors.to_rgba('#70CDBE'), alpha=0.2,
                 zorder=2)
bax.fill_between(index_MOBO50 + 160, np.mean(RHD_MOBO50, axis=0) - std_RHD_MOBO50,
                 np.mean(RHD_MOBO50, axis=0) + std_RHD_MOBO50, color=mcolors.to_rgba('#F9B3AD'), alpha=0.2,
                 zorder=3)
bax.fill_between(index_NSGA2 + 160, np.mean(RHD_NSGA2, axis=0) - std_RHD_NSGA2,
                 np.mean(RHD_NSGA2, axis=0) + std_RHD_NSGA2, color=mcolors.to_rgba('#FFDD8E'), alpha=0.2,
                 zorder=1)
bax.fill_between(index_NB2 + 160, np.mean(RHD_TSO2, axis=0) - std_RHD_NB2,
                 np.mean(RHD_TSO2, axis=0) + std_RHD_NB2, color=mcolors.to_rgba('#8FB4DC'), alpha=0.2,
                 zorder=4)
bax.fill_between(index_XLO2 + 160, np.mean(RHD_XLO2, axis=0) - std_RHD_XLO2,
                 np.mean(RHD_XLO2, axis=0) + std_RHD_XLO2, color=mcolors.to_rgba('#AC99D2'), alpha=0.2,
                 zorder=5)

bax.set_xlabel('Number of sample point evaluations', fontsize=28, labelpad=35, fontname='Arial')
bax.set_ylabel('Relative hypervolume ratio', fontsize=28, labelpad=53, fontname='Arial')
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
    right=0.97
)

plt.savefig('RHR Reactor.svg', dpi=800)
plt.show()