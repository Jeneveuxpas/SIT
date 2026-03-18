import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# 均匀间距 x 轴（正常刻度）
steps = [1, 2, 3]
x_labels = ['100K', '200K', '400K']

# SiT-XL/2
repa_fid            = [20.1, 13.1, 11.1]
irepa_fid           = [16.8, 11.7, 10.3]
attn_scaffolding_fid = [12.80, 8.47, 7.09]

# 配色（与原文件一致）
color_repa = '#9ed4f6'
color_irepa = '#FFA500'
color_ours  = '#FF4040'

fig, ax = plt.subplots(figsize=(7.2, 6.6))

light_alpha = 0.65

ax.plot(steps, repa_fid, color=color_repa, marker='o', markersize=12,
        markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
        alpha=light_alpha, label='SiT-XL/2 + REPA', zorder=2)

ax.plot(steps, irepa_fid, color=color_irepa, marker='o', markersize=12,
        markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
        alpha=light_alpha, label='SiT-XL/2 + iREPA', zorder=2)

ax.plot(steps, attn_scaffolding_fid, color=color_ours, marker='o', markersize=14,
        markerfacecolor='white', markeredgewidth=3.0, linewidth=9.8,
        alpha=0.85, label='SiT-XL/2 + AttnScaf', zorder=5)

# ── 400K 终点虚线 ──
ax.axhline(y=repa_fid[-1],             color=color_repa,  linestyle='--', linewidth=1.9, alpha=0.85)
ax.axhline(y=irepa_fid[-1],            color=color_irepa, linestyle='--', linewidth=1.9, alpha=0.85)
ax.axhline(y=attn_scaffolding_fid[-1], color=color_ours,  linestyle='--', linewidth=1.9, alpha=0.85)

ax.annotate(f'{repa_fid[-1]}', xy=(3, repa_fid[-1]),
            xytext=(2.55, repa_fid[-1] * 1.022),
            fontsize=13.3, fontweight='bold', color='#157BF8', ha='left', va='center', alpha=0.95)
ax.annotate(f'{irepa_fid[-1]}', xy=(3, irepa_fid[-1]),
            xytext=(2.55, irepa_fid[-1] * 0.95),
            fontsize=13.3, fontweight='bold', color='#FFA500', ha='left', va='center', alpha=0.95)
ax.annotate(f'{attn_scaffolding_fid[-1]}', xy=(3, attn_scaffolding_fid[-1]),
            xytext=(2.55, attn_scaffolding_fid[-1] * 0.95),
            fontsize=13.3, fontweight='bold', color='#FF2800', ha='left', va='center', alpha=0.95)


# ── Y 轴线性刻度 ──
ax.set_yscale('linear')
ax.set_yticks([7.09, 10.3, 11.1, 20])
ax.set_yticklabels(['7.09', '10.3', '11.1', '20'], fontsize=14.5)

# ── X 轴（均匀间距） ──
ax.set_xlim(0.8, 3.2)
ax.set_xticks(steps)
ax.set_xticklabels(x_labels, fontsize=15)

ax.set_xlabel('Training Steps', fontsize=17.5, labelpad=8)
ax.set_ylabel(r'FID Score $\downarrow$', fontsize=17.5, labelpad=8)
ax.tick_params(axis='both', which='major', length=6, width=1.25)

# ── 网格 ──
ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.05, color='#c9ccd1')
ax.set_axisbelow(True)

# ── 边框 ──
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.45)
ax.spines['bottom'].set_linewidth(1.45)

# ── 图例 ──
legend = ax.legend(loc='upper right', frameon=True, fontsize=16)
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.2)
legend.get_frame().set_alpha(0.95)

ax.minorticks_off()

for text, handle in zip(legend.get_texts(), legend.legend_handles):
    text.set_color(handle.get_color())
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('fid_chart_linear.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fid_chart_linear.png', dpi=300, bbox_inches='tight')
print("Done!")
