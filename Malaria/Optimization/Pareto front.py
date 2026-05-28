import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

PF_NSGA23 = []
for j in range(10):
    PF_NSGA23.append(np.load(os.path.join(root_dir,
        'NSGA23\Results\PF-NSGA23-Malaria-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA23 = np.array(np.concatenate(PF_NSGA23, axis=0))

PF_MOBO23 = []
for j in range(10):
    PF_MOBO23.append(np.load(os.path.join(root_dir,
        'MOBO23\Results\PF-MOBO23-Malaria-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO23 = np.array(np.concatenate(PF_MOBO23, axis=0))

PF_NSGA5 = []
for j in range(10):
    PF_NSGA5.append(np.load(os.path.join(root_dir,
        'NSGA5\Results\PF-NSGA5-Malaria-repetition-' + str(
            j) + '.npy')))

all_PF_NSGA5 = np.concatenate(PF_NSGA5, axis=0)

PF_MOBO5 = []
for j in range(10):
    PF_MOBO5.append(np.load(os.path.join(root_dir,
        'MOBO5\Results\PF-MOBO5-Malaria-repetition-' + str(
            j) + '.npy')))

all_PF_MOBO5 = np.concatenate(PF_MOBO5, axis=0)

PF_XLO5 = []
for j in range(10):
    PF_XLO5.append(np.load(os.path.join(root_dir,
        'XLO\Results\PF-XLO-Malaria-repetition-' + str(
            j) + '.npy')))

all_PF_XLO5 = np.concatenate(PF_XLO5, axis=0)

all_PF_NSGA23 = np.unique(all_PF_NSGA23, axis=0)
all_PF_MOBO23 = np.unique(all_PF_MOBO23, axis=0)
all_PF_NSGA5 = np.unique(all_PF_NSGA5, axis=0)
all_PF_MOBO5 = np.unique(all_PF_MOBO5, axis=0)
all_PF_XLO5 = np.unique(all_PF_XLO5, axis=0)

fig, ax = plt.subplots(figsize=(14, 8))

ax.scatter(all_PF_NSGA23[:, 0], all_PF_NSGA23[:, 1],
           s=200, color=mcolors.to_rgba('#70CDBE'), edgecolor='k', alpha=1, linewidths=0.8, label='NSGA-II (23)', marker='^', zorder=1)
ax.scatter(all_PF_MOBO23[:, 0], all_PF_MOBO23[:, 1],
           s=200, color=mcolors.to_rgba('#F9B3AD'), edgecolor='k', alpha=1, linewidths=0.8, label='MOBO (23)', marker='o')
ax.scatter(all_PF_NSGA5[:, 0], all_PF_NSGA5[:, 1],
           s=200, color=mcolors.to_rgba('#FFDD8E'), edgecolor='k', alpha=1, linewidths=0.8, label='NSGA-II (5)', marker='v')
ax.scatter(all_PF_MOBO5[:, 0], all_PF_MOBO5[:, 1],
           s=200, color=mcolors.to_rgba('#8FB4DC'), edgecolor='k', alpha=1, linewidths=0.8, label='MOBO (5)', marker='s')
ax.scatter(all_PF_XLO5[:, 0], all_PF_XLO5[:, 1],
           s=450, color=mcolors.to_rgba('#AC99D2'), edgecolor='k', alpha=1, linewidths=0.8, label='XLO (5)', marker='*')

ax.set_xlabel(r'$\ell_{\mathrm{Bin}}(\mathbf{p})$', labelpad=10, fontsize=28, fontname='Arial')
ax.set_ylabel(r'$\ell_{\mathrm{LN}}(\mathbf{d})$', labelpad=10, fontsize=28, fontname='Arial')
ax.tick_params(axis='both', labelsize=22, length=8, width=2.5)

tick_font = FontProperties(family='Arial', size=22)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

ax.tick_params(labelsize=20)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontproperties(tick_font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

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
plt.savefig('PF Malaria.svg', dpi=800)

plt.show()