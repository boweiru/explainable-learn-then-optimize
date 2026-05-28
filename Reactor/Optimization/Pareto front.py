import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

PF_NSGA50 = []
for j in range(5):
    PF_NSGA50.append(np.load(os.path.join(root_dir,
        'NSGA50\Results\PF-NSGA50-Reactor-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA50 = np.array(np.concatenate(PF_NSGA50, axis=0))

PF_MOBO50 = []
for j in range(5):
    PF_MOBO50.append(np.load(os.path.join(root_dir,
        'MOBO50\Results\PF-MOBO50-Reactor-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO50 = np.array(np.concatenate(PF_MOBO50, axis=0))

PF_NSGA2 = []
for j in range(5):
    PF_NSGA2.append(np.load(os.path.join(root_dir,
        'NSGA2\Results\PF-NSGA2-Reactor-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA2 = np.concatenate(PF_NSGA2, axis=0)

PF_MOBO2 = []
for j in range(5):
    PF_MOBO2.append(np.load(os.path.join(root_dir,
        'MOBO2\Results\PF-MOBO2-Reactor-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO2 = np.concatenate(PF_MOBO2, axis=0)

PF_XLO2 = []
for j in range(5):
    PF_XLO2.append(np.load(os.path.join(root_dir,
        'XLO\Results\PF-XLO-Reactor-repetition-' + str(
            j) + '.npy')))

all_PF_XLO2 = np.concatenate(PF_XLO2, axis=0)

all_PF_NSGA50 = np.unique(all_PF_NSGA50, axis=0)
all_PF_MOBO50 = np.unique(all_PF_MOBO50, axis=0)
all_PF_NSGA2 = np.unique(all_PF_NSGA2, axis=0)
all_PF_MOBO2 = np.unique(all_PF_MOBO2, axis=0)
all_PF_XLO2 = np.unique(all_PF_XLO2, axis=0)

fig, ax = plt.subplots(figsize=(14, 8))

ax.scatter(all_PF_NSGA50[:, 0], all_PF_NSGA50[:, 1],
           s=200, color=mcolors.to_rgba('#70CDBE'), edgecolor='k', alpha=1, linewidths=0.8, label='NSGA-II (50)', marker='^', zorder=1)
ax.scatter(all_PF_MOBO50[:, 0], all_PF_MOBO50[:, 1],
           s=200, color=mcolors.to_rgba('#F9B3AD'), edgecolor='k', alpha=1, linewidths=0.8, label='MOBO (50)', marker='o')
ax.scatter(all_PF_NSGA2[:, 0], all_PF_NSGA2[:, 1],
           s=200, color=mcolors.to_rgba('#FFDD8E'), edgecolor='k', alpha=1, linewidths=0.8, label='NSGA-II (2)', marker='v')
ax.scatter(all_PF_MOBO2[:, 0], all_PF_MOBO2[:, 1],
           s=200, color=mcolors.to_rgba('#8FB4DC'), edgecolor='k', alpha=1, linewidths=0.8, label='MOBO (2)', marker='s')
ax.scatter(all_PF_XLO2[:, 0], all_PF_XLO2[:, 1],
           s=450, color=mcolors.to_rgba('#AC99D2'), edgecolor='k', alpha=1, linewidths=0.8, label='XLO (2)', marker='*')

ax.set_xlabel(r'$N^{*}$', labelpad=10, fontsize=28, fontname='Arial')
ax.set_ylabel('MSE', labelpad=10, fontsize=28, fontname='Arial')
tick_font = FontProperties(family='Arial', size=20)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

ax.tick_params(labelsize=20)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontproperties(tick_font)

ax.tick_params(axis='both', labelsize=22, length=8, width=2.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

font_prop = FontProperties(
    family='Arial',
    size=28,
    weight='normal'
)

legend = ax.legend(
    loc='upper right',
    frameon=True,
    edgecolor='black',
    prop=font_prop,
    handletextpad=0.5,
    borderaxespad=0.5,
    labelspacing=0.3,
    markerscale=1.3
)
legend.get_frame().set_linewidth(2.5)

fig.tight_layout()
plt.savefig('PF Reactor.svg', dpi=800)

plt.show()