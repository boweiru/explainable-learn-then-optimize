import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

def create_ball_image(color='#4C9F89', size=40):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, size-1, size-1], fill=color)
    highlight = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw_highlight = ImageDraw.Draw(highlight)
    draw_highlight.ellipse([size*0.15, size*0.15, size*0.55, size*0.55], fill=(255, 255, 255, 100))
    img = Image.alpha_composite(img, highlight)
    return img

ball_img = create_ball_image()

tmp_path = 'temp_ball.png'
ball_img.save(tmp_path)

PT = np.load('PT-convergence.npy')
k = 20
sum_PT = np.zeros(k)
for i in range(k):
    sum_PT[i] = np.sum(PT[i, :])
Relative_change = np.zeros(k - 1)
for i in range(k - 1):
    Relative_change[i] = abs(sum_PT[i + 1] - sum_PT[i]) / sum_PT[i]

x_best = 12
y_best = sum_PT[x_best - 1]
sum_PT = sum_PT[4:]
x = np.arange(4, 20)

fig, ax = plt.subplots(figsize=(12, 7))


ax.plot(x, sum_PT, linestyle='-', color='#4C9F89', linewidth=5, zorder=1)

ball_arr = plt.imread(tmp_path)
imagebox = OffsetImage(ball_arr, zoom=0.4)

for xi, yi in zip(x, sum_PT):
    ab = AnnotationBbox(imagebox, (xi, yi), frameon=False, pad=0.0, zorder=3)
    ax.add_artist(ab)

ax.plot(x_best - 1, y_best, marker='*', markersize=40,
        markerfacecolor='none', markeredgecolor='#7A66B9',
        markeredgewidth=2, linestyle='', zorder=4)

ax.set_xlabel('Monte Carlo sample size', fontsize=27, fontname='Arial', labelpad=10)
ax.set_ylabel('Convergence curve', fontsize=27, fontname='Arial', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(x + 1, ha='center', fontsize=24, fontname='Arial')
ax.tick_params(axis='both', labelsize=24, length=8, width=2.5)
ax.text(1, -0.02, r'$2^x$', transform=ax.transAxes, fontsize=24,
        verticalalignment='top', horizontalalignment='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

class HandlerBallImage(HandlerBase):
    def __init__(self, image, zoom=0.4):
        super().__init__()
        self.image = image
        self.zoom = zoom

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        oi = OffsetImage(self.image, zoom=self.zoom)
        ab = AnnotationBbox(oi, (xdescent + width / 2, ydescent + height / 2),
                            frameon=False, box_alignment=(0.5, 0.5), pad=0)
        ab.set_transform(trans)
        return [ab]

legend_x = 0.725
legend_y = 0.8

legend_box = FancyBboxPatch(
    (legend_x - 0.07, legend_y - 0.074),
    0.272, 0.12,
    transform=ax.transAxes,
    boxstyle="round,pad=0.02,rounding_size=0.015",
    linewidth=2.5,
    edgecolor='black',
    facecolor='white',
    zorder=4
)
ax.add_patch(legend_box)

line = Line2D(
    [legend_x - 0.07, legend_x + 0.07],
    [legend_y, legend_y],
    transform=ax.transAxes,
    color='#4C9F89',
    linewidth=5,
    zorder=6
)
ax.add_line(line)

ab = AnnotationBbox(
    OffsetImage(ball_arr, zoom=0.5),
    (legend_x, legend_y),  # 中间
    frameon=False,
    xycoords='axes fraction',
    box_alignment=(0.5, 0.5),
    pad=0.0,
    zorder=6
)
ax.add_artist(ab)

ax.text(legend_x + 0.085, legend_y-0.02,
        r'$\sum_{i} \mathit{\hat{P}_{i}^{\mathrm{total}}}$',
        transform=ax.transAxes, fontsize=27, va='center', ha='left', zorder=7)

fig.canvas.draw()
y0, y1 = ax.get_ylim()
frac = (y_best - y0) / (y1 - y0)
ax.axvline(x=x_best - 1, ymin=0, ymax=frac, ls='--', color='gray', lw=2.5)

plt.tight_layout()
plt.savefig('Monte Carlo Corridor 3Dball.svg',
            dpi=800, bbox_inches='tight')
plt.show()






