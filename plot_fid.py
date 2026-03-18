import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置全局样式：统一放大文字与线条，避免图面偏空
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# 准备数据 —— 拉大 100K 到 200K 的间距
steps = [1, 3.2, 6.0]
x_labels = ['100K', '200K', '400K']

# SiT-XL/2
repa_fid = [19.4, 11.1, 7.9]
irepa_fid = [16.9, 10.3, 7.52]
haste_fid = [15.9, 9.9, 7.3]
attn_scaffolding_fid = [11.6, 7.9, 6.8]

# # SiT-L/2
# repa_fid = [24.1, 14.0, 10.0]
# irepa_fid = [20.28,12.92, 9.36]
# haste_fid = [19.6, 12.1, 8.9]
# attn_scaffolding_fid = [15.1, 10.1, 8.25]

# # SiT-B/2
# repa_fid = [49.5, 33.2, 24.4]
# irepa_fid = [38.5, , 21.40]
# haste_fid = [39.9, 25.7,19.6]
# attn_scaffolding_fid = [11.6, 8.2, 6.8]

# 配色
color_repa = '#9ed4f6'
color_irepa = '#FFA500'
color_haste = '#83C86E'
color_ours = '#FF4040'

fig, ax = plt.subplots(figsize=(7.2, 6.6))

# ── 其他方法：颜色浅 ──
light_alpha = 0.65
ax.plot(steps, repa_fid, color=color_repa, marker='o', markersize=12,
        markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
        alpha=light_alpha, label='SiT-XL/2 + REPA', zorder=2)
ax.plot(steps, irepa_fid, color=color_irepa, marker='o', markersize=12,
        markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
        alpha=light_alpha, label='SiT-XL/2 + iREPA', zorder=2)
ax.plot(steps, haste_fid, color=color_haste, marker='o', markersize=12,
        markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
        alpha=light_alpha, label='SiT-XL/2 + HASTE', zorder=2)

# ── 我的方法：深色、加粗 ──
ax.plot(steps, attn_scaffolding_fid, color=color_ours, marker='o', markersize=14,
        markerfacecolor='white', markeredgewidth=3.0, linewidth=9.8,
        alpha=0.85,label='SiT-XL/2 + Attn. Scaff.', zorder=5)

# ── 只画 400K 时的虚线 ──
ax.axhline(y=7.9, color=color_repa, linestyle='--', linewidth=1.9, alpha=0.85)
ax.axhline(y=6.8, color=color_ours, linestyle='--', linewidth=1.9, alpha=0.85)

ax.annotate('7.9', xy=(1, 7.9), xytext=(2.3, 8.08),
            fontsize=13.3, fontweight='bold', color='#157BF8', ha='left', va='center',alpha=0.95)
ax.annotate('6.8', xy=(1, 6.8), xytext=(2.3, 6.63),
            fontsize=13.3, fontweight='bold', color='#FF2800', ha='left', va='center',alpha=0.95)
# ── 100K 处纵向灰色参考线 ──
ax.axvline(x=1, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=1.0, zorder=1)


# ax.axhline(y=15.9,color='gray',linewidth=0.8,linestyle='--',alpha=0.2,zorder=1)  
ax.axhline(y=19.4,color='gray',linewidth=0.9,linestyle='--',alpha=0.5,zorder=1) 
ax.axhline(y=11.6,color='gray',linewidth=0.9,linestyle='--',alpha=0.5,zorder=1) 
# ── 100K 处标注全部 4 个方法的 FID 值（放在点的右侧） ──
annot_x = 1.18  # 标注的 x 偏移
ax.annotate('19.4', xy=(1, 19.4), xytext=(annot_x, 18.98),
            fontsize=15.4, fontweight='bold', color='#157BF8', ha='left', va='center')
ax.annotate('16.9', xy=(1, 16.9), xytext=(annot_x, 16.443),
            fontsize=15.4, fontweight='bold', color='#ff7710', ha='left', va='center')
ax.annotate('15.9', xy=(1, 15.9), xytext=(annot_x, 15.443),
            fontsize=15.4, fontweight='bold', color='#109D59', ha='left', va='center')
ax.annotate('11.6', xy=(1, 11.6), xytext=(annot_x, 11.85),
            fontsize=16.0, fontweight='bold', color='#FF2800', ha='left', va='center')

# ── 坐标轴 ──
ax.set_yscale('log')
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_yticks([6.8, 7.9, 10, 20])
ax.set_yticklabels(['6.8', '7.9', '10', '20'], fontsize=14.5)

ax.set_xlim(0.8, 6.2)
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
# 图例文字颜色与对应曲线颜色一致
for text, handle in zip(legend.get_texts(), legend.legend_handles):
    text.set_color(handle.get_color())
for text in legend.get_texts():
    text.set_fontweight('bold')

# 保存
plt.tight_layout()
plt.savefig('fid_chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fid_chart.png', dpi=300, bbox_inches='tight')
print("Done!")
