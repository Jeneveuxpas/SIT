import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 全局样式 ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'legend.fontsize': 19,
})

# ── 数据 ──
steps = [1, 3.2, 6.0]
x_labels = ['100K', '200K', '400K']

# 各规模数据: (REPA, iREPA, HASTE, AttnScaf)
data = {
    'SiT-B/2': {
        'REPA':       [49.5, 33.2, 24.4],
        'iREPA':      [38.5, 27.45, 21.40],
        'HASTE':      [39.9, 25.7, 19.6],
        'AttnScaf': [33.0, 23.48, 19.26],
    },
    'SiT-L/2': {
        'REPA':       [24.1, 14.0, 10.0],
        'iREPA':      [20.28, 12.92, 9.36],
        'HASTE':      [19.6, 12.1, 8.9],
        'AttnScaf': [15.1, 10.1, 8.25],
    },
    'SiT-XL/2': {
        'REPA':       [19.4, 11.1, 7.9],
        'iREPA':      [16.9, 10.3, 7.52],
        'HASTE':      [15.9, 9.9, 7.3],
        'AttnScaf': [11.6, 7.9, 6.8],
    },
}

# ── 配色 ──
colors = {
    'REPA':       '#9ed4f6',
    'iREPA':      '#FFA500',
    'HASTE':      '#83C86E',
    'AttnScaf': '#FF0000',
}
dark_colors = {
    'REPA':       '#157BF8',
    'iREPA':      '#ff7710',
    'HASTE':      '#109D59',
    'AttnScaf': '#FF0000',
}

panel_order = ['SiT-B/2', 'SiT-L/2', 'SiT-XL/2']
panel_tags  = ['b', 'l', 'xl']

for idx, (model_name, tag) in enumerate(zip(panel_order, panel_tags)):
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    model_data = data[model_name]
    light_alpha = 0.65

    # ── 画其他方法 ──
    for method in ['REPA', 'iREPA', 'HASTE']:
        fid_vals = model_data[method]
        valid = [(s, f) for s, f in zip(steps, fid_vals) if f is not None]
        if not valid:
            continue
        s_valid, f_valid = zip(*valid)
        ax.plot(s_valid, f_valid, color=colors[method], marker='o', markersize=12,
                markerfacecolor='white', markeredgewidth=2.1, linewidth=8.2,
                alpha=light_alpha, label=f'{model_name} + {method}', zorder=2)

    # ── 画 Ours ──
    ours_fid = model_data['AttnScaf']
    ax.plot(steps, ours_fid, color=colors['AttnScaf'], marker='o', markersize=15,
            markerfacecolor='white', markeredgewidth=3.2, linewidth=10.5,
            alpha=0.89, label=f'{model_name} + AttnScaf', zorder=5)

    # ── 400K 虚线对比: REPA vs Ours ──
    repa_400k = model_data['REPA'][-1]
    ours_400k = model_data['AttnScaf'][-1]
    ax.axhline(y=repa_400k, color=colors['REPA'], linestyle='--', linewidth=1.9, alpha=0.85)
    ax.axhline(y=ours_400k, color=colors['AttnScaf'], linestyle='--', linewidth=1.9, alpha=0.85)

    # ── 400K 虚线标注 (per-panel 配置) ──
    # 格式: (x, y_offset_repa, y_offset_ours)
    #   x: 文字的 x 坐标
    #   y_offset_repa: REPA 标注相对虚线的 y 偏移 (正=上, 负=下)
    #   y_offset_ours: AttnScaf 标注相对虚线的 y 偏移
    annot_400k_cfg = {
        'SiT-B/2':  {'x': 1.8, 'repa_dy': +0.25, 'ours_dy': -0.1},
        'SiT-L/2':  {'x': 2.0, 'repa_dy': +0.12, 'ours_dy': -0.15},
        'SiT-XL/2': {'x': 2.0, 'repa_dy': +0.1, 'ours_dy': -0.15},
    }
    cfg = annot_400k_cfg[model_name]
    ax.annotate(f'{repa_400k}', xy=(steps[-1], repa_400k),
                xytext=(cfg['x'], repa_400k + cfg['repa_dy']),
                fontsize=19, fontweight='bold', color=dark_colors['REPA'],
                ha='left', va='bottom', alpha=0.95)
    ax.annotate(f'{ours_400k}', xy=(steps[-1], ours_400k),
                xytext=(cfg['x'], ours_400k + cfg['ours_dy']),
                fontsize=20, fontweight='bold', color=dark_colors['AttnScaf'],
                ha='left', va='top', alpha=1.0)

    # ── 100K 处纵向灰色参考线 ──
    ax.axvline(x=1, color='#aaaaaa', linestyle=':', linewidth=1.2, alpha=1.0, zorder=1)

    # ── 100K 处水平参考线 ──
    repa_100k = model_data['REPA'][0]
    ours_100k = model_data['AttnScaf'][0]
    ax.axhline(y=repa_100k, color='gray', linewidth=0.9, linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=ours_100k, color='gray', linewidth=0.9, linestyle='--', alpha=0.5, zorder=1)

    # ── 100K 标注所有方法的 FID (per-panel 配置) ──
    # 格式: {方法名: {'x': x坐标, 'dy': y偏移}}
    # dy 是相对数据点的偏移 (正=上, 负=下)
    annot_100k_cfg = {
        'SiT-B/2': {
            'REPA':     {'x': 1.18, 'dy': 0},
            'iREPA':    {'x': 1.18, 'dy': -0.3},
            'HASTE':    {'x': 1.18, 'dy': 0.3},
            'AttnScaf': {'x': 1.18, 'dy': 0},
        },
        'SiT-L/2': {
            'REPA':     {'x': 1.18, 'dy': 0},
            'iREPA':    {'x': 1.18, 'dy': 0.1},
            'HASTE':    {'x': 1.18, 'dy': -0.5},
            'AttnScaf': {'x': 1.18, 'dy': 0},
        },
        'SiT-XL/2': {
            'REPA':     {'x': 1.18, 'dy': 0},
            'iREPA':    {'x': 1.18, 'dy': -0.2},
            'HASTE':    {'x': 1.18, 'dy': -0.5},
            'AttnScaf': {'x': 1.18, 'dy': 0},
        },
    }
    cfg_100k = annot_100k_cfg[model_name]
    for method_name in ['REPA', 'iREPA', 'HASTE', 'AttnScaf']:
        val = model_data[method_name][0]
        if val is None:
            continue
        mc = cfg_100k[method_name]
        fs = 20.0 if method_name == 'AttnScaf' else 19.0
        ax.annotate(f'{val}', xy=(1, val),
                    xytext=(mc['x'], val + mc['dy']),
                    fontsize=fs, fontweight='bold', color=dark_colors[method_name],
                    ha='left', va='center')

    # ── 坐标轴: 用 log 坐标 + 精调 ylim ──
    # SiT-B/2 数据范围大(19~50)，底部 400K 处差异小，
    # 用分段缩放: 低区线性展开，高区 log 压缩
    if model_name == 'SiT-B/2':
        bp, k =19.5, 4.0
        def _fwd(x, _bp=bp, _k=k):
            x = np.asarray(x, dtype=float)
            return np.where(x <= _bp, x, _bp + np.log(x / _bp) * _k)
        def _inv(y, _bp=bp, _k=k):
            y = np.asarray(y, dtype=float)
            return np.where(y <= _bp, y, _bp * np.exp((y - _bp) / _k))
        ax.set_yscale('function', functions=(_fwd, _inv))
    else:
        ax.set_yscale('log')
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.minorticks_off()

    # 精调 ylim：紧贴数据，只留少量 padding
    all_vals = []
    for method_vals in model_data.values():
        all_vals.extend([v for v in method_vals if v is not None])
    y_min = min(all_vals) * (0.975 if model_name == 'SiT-B/2' else 0.88)
    y_max = max(all_vals) * 1.08
    ax.set_ylim(y_min, y_max)

    # 设置 yticks
    nice_ticks = sorted(set([round(v, 1) for v in [min(all_vals), max(all_vals),
                                                     repa_400k, ours_400k]]))
    mid = (min(all_vals) + max(all_vals)) / 2
    nice_ticks.append(round(mid, 0))
    nice_ticks = sorted(set(nice_ticks))
    ax.set_yticks(nice_ticks)
    ax.set_yticklabels([f'{v:g}' for v in nice_ticks], fontsize=19)

    ax.set_xlim(0.8, 6.2)
    ax.set_xticks(steps)
    ax.set_xticklabels(x_labels, fontsize=19)

    ylabel_pad = 4 if model_name in ('SiT-L/2', 'SiT-XL/2') else 8
    ax.set_xlabel('Training Steps', fontsize=22, labelpad=8)
    ax.set_ylabel(r'FID Score $\downarrow$', fontsize=22, labelpad=ylabel_pad)
    ax.tick_params(axis='both', which='major', length=6, width=1.25)

    # ── 网格 ──
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.05, color='#c9ccd1')
    ax.set_axisbelow(True)

    # ── 边框 ──
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.45)
    ax.spines['bottom'].set_linewidth(1.45)

    # ── 子图标题 ──
    ax.set_title(model_name, fontsize=24, pad=12)

    # ── 图例 ──
    legend = ax.legend(loc='upper right', frameon=True, fontsize=19, labelspacing=0.3, borderpad=0.38)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.95)

    method_names = ['REPA', 'iREPA', 'HASTE', 'AttnScaf']
    for text, mname in zip(legend.get_texts(), method_names):
        text.set_color(dark_colors[mname])
        if mname == 'AttnScaf':
            text.set_fontweight('bold')
        else:
            text.set_fontweight('normal')

    # ── 保存单独文件 ──
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out = f'fid_chart_{tag}.{ext}'
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved fid_chart_{tag}.pdf/png')

# ── 拼合成一张横排大图 ──
from PIL import Image

imgs = [Image.open(f'fid_chart_{t}.png') for t in panel_tags]
# 统一高度
max_h = max(im.height for im in imgs)
imgs_resized = []
for im in imgs:
    if im.height != max_h:
        ratio = max_h / im.height
        im = im.resize((int(im.width * ratio), max_h), Image.LANCZOS)
    imgs_resized.append(im)

total_w = sum(im.width for im in imgs_resized)
combined = Image.new('RGB', (total_w, max_h), 'white')
x_offset = 0
for im in imgs_resized:
    combined.paste(im, (x_offset, 0))
    x_offset += im.width

combined.save('fid_chart_3panel.png', dpi=(300, 300))

# 也保存为 PDF
combined.save('fid_chart_3panel.pdf', dpi=(300, 300))

print("\nDone! Individual + combined fid_chart_3panel.pdf/png saved.")
