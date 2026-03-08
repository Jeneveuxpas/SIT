import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- data ----------
methods = ['SiT-B/2', '+REPA', '+HASTE', '+iREPA','+AttnScaf']
clip_fid = [None, 54.9, 48.7, 41.8, 36.2]
dino_fid = [None, 49.5, 39.9, 38.5, 33.0]
baseline = 63.46

x = np.arange(len(methods))
bar_width = 0.36

# ---------- style ----------
color_clip = '#2f81b7'
color_dino = '#c9211a'

color_base = '#aaa9a9'
color_text = '#000000'

color_value_base = '#000000'

color_value_clip = '#000000'
color_value_dino = '#000000'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'text.color': color_text,
    'axes.labelcolor': color_text,
    'xtick.color': color_text,
    'ytick.color': color_text,
})

fig, ax = plt.subplots(figsize=(7.5, 3.8))

# Baseline bar (centered, hatched)
ax.bar(x[0], baseline, width=bar_width * 1.8, color=color_base,
       alpha=0.55, hatch='///', edgecolor='#bbb', linewidth=0.6, zorder=3)
ax.text(x[0], baseline + 1, f'{baseline}', ha='center', va='bottom',
        fontsize=9, fontweight='normal', color=color_value_base)

# CLIP bars (left)
clip_vals = [v for v in clip_fid if v is not None]
ax.bar(x[1:] - bar_width/2, clip_vals, width=bar_width,
       color=color_clip, edgecolor='white', linewidth=0.8,
       alpha=0.9, zorder=3)
for xi, v in zip(x[1:], clip_vals):
    ax.text(xi - bar_width/2, v + 1, f'{v}', ha='center', va='bottom',
            fontsize=9, fontweight='normal', color=color_value_clip)

# DINOv2 bars (right)
dino_vals = [v for v in dino_fid if v is not None]
ax.bar(x[1:] + bar_width/2, dino_vals, width=bar_width,
       color=color_dino, edgecolor='white', linewidth=0.8,
       alpha=0.85, zorder=3)
for xi, v in zip(x[1:], dino_vals):
    ax.text(xi + bar_width/2, v + 1, f'{v}', ha='center', va='bottom',
            fontsize=9, fontweight='normal', color=color_value_dino)

# Baseline dashed line
ax.axhline(y=baseline, color='#bbb', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)

# ---------- axes ----------
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('FID $\\downarrow$', fontsize=12)
ax.set_ylim(0, 74)
ax.set_xlim(-0.5, len(methods) - 0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['left'].set_linewidth(1.0)

# Force x-axis to draw on top of bars
for spine in ax.spines.values():
    spine.set_zorder(10)
ax.yaxis.grid(True, linestyle='-', alpha=0.12, zorder=0)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_base, hatch='///', alpha=0.55,
                   edgecolor='#bbb', label='Vanilla'),
    mpatches.Patch(facecolor=color_clip, alpha=0.9, label='CLIP-ViT-B'),
    mpatches.Patch(facecolor=color_dino, alpha=0.9, label='DINOv2-B'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True,
          framealpha=0.95, edgecolor='#ddd', fontsize=9.5)

plt.tight_layout()
plt.savefig('encoder.pdf', dpi=300, bbox_inches='tight')
plt.savefig('encoder.png', dpi=300, bbox_inches='tight')
plt.close()
print('Done!')
