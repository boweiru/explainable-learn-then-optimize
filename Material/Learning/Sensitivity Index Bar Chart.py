import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

num_vars_all = 27

names_all = ['x' + str(i + 1) for i in range(num_vars_all)]
names_all = np.array(names_all)

PT = np.load('PT.npy')

sorted_idx = np.argsort(-PT)
names_all = names_all[sorted_idx]
PT = PT[sorted_idx]

numbers = 27
sorted_idx = sorted_idx[:numbers]
names_all = names_all[:numbers]
PT = PT[:numbers]

cum = np.load('cum_PT_descending.npy')[:numbers]


bar_width = 0.7
x = np.arange(numbers)

fig, ax1 = plt.subplots(figsize=(18, 10))

bars = ax1.bar(x, PT, color=mcolors.to_rgba('#C6DCF0'), edgecolor=mcolors.to_rgba('#4A8EBB'),
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

ax2.set_ylabel('Cumulative variance contribution', fontsize=35, fontname='Arial', labelpad=10)
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
plt.savefig('Sensitivity Material.svg', dpi=800,
            bbox_inches='tight')
plt.show()

