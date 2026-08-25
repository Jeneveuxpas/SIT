"""Plot the four FID comparisons with an independently scaled y-axis per panel."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── 全局样式 ──
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

# 三个 training step 使用等间距，避免 200K→400K 在视觉上是两倍距离。
steps = np.array([1.1, 2.0, 3.5])
x_labels = ['100K', '200K', '400K']

# None 表示该实验尚未在该 iteration 测量；画图时会自动跳过。
data = {
    'SiT-B/2': {
        'REPA': [49.50, 33.20, 24.40],
        'REPI': [35.90, 25.00, 19.69],
        'REPI + REPA': [35.81, 23.97, 18.59],
    },
    'SiT-L/2': {
        'REPA': [24.10, 14.00, 10.00],
        'REPI': [19.02, 12.08, 9.43],
        'REPI + REPA': [14.71, 10.07, 8.16],
    },
    'SIT-XL/2': {
        'REPA': [19.4, 11.1, 7.9],
        'REPI': [14.69, 9.16, 7.41],
        'REPI + REPA': [11.85, 7.47, 6.38],
    },
    'DiT-L/2': {
        'REPA': [32.90, 20.37, 14.50],
        'REPI': [26.01, 17.80, 13.59],
        'REPI + REPA': [25.04, 16.10, 12.24],
    },
}

panel_order = ['SiT-B/2', 'SiT-L/2', 'SIT-XL/2', 'DiT-L/2']
panel_tags = ['sit_b', 'sit_l', 'sit_xl', 'dit_l']
method_order = ['REPA', 'REPI', 'REPI + REPA']

color_repa = '#9ed4f6'
color_repi = '#FFA500'
color_ours = '#FF4040'

colors = {
    'REPA': color_repa,
    'REPI': color_repi,
    'REPI + REPA': color_ours,
}
line_styles = {'REPA': '-', 'REPI': '-', 'REPI + REPA': '-'}
markers = {'REPA': 'o', 'REPI': 'o', 'REPI + REPA': 'o'}

panel_yaxis = {
    'SiT-B/2': {'ylim': (16.0, 54.0), 'ticks': [20, 30, 40, 50]},
    'SiT-L/2': {'ylim': (7.0, 26.0), 'ticks': [8, 12, 16, 20, 24]},
    'SIT-XL/2': {'ylim': (5.8, 22.0), 'ticks': [8, 12, 16, 20]},
    'DiT-L/2': {'ylim': (11.0, 35.0), 'ticks': [12, 16, 20, 24, 28, 32]},
}

# 终值标签相对虚线的纵向偏移，避免相邻数字和曲线重叠。
final_label_offsets = {
    'SiT-B/2': {'REPA': 0.70, 'REPI': 0.69, 'REPI + REPA': -0.79},
    'SiT-L/2': {'REPA': 0.45, 'REPI': -0.40, 'REPI + REPA': -0.40},
    'SIT-XL/2': {'REPA': 0.45, 'REPI': 0.21, 'REPI + REPA': 0.35},
    'DiT-L/2': {'REPA': 0.40, 'REPI': -0.55, 'REPI + REPA': -0.55},
}


def finite_points(values):
    """Return the measured x/y pairs, ignoring missing measurements."""
    return [(step, value) for step, value in zip(steps, values) if value is not None]


def draw_final_guides(ax, model_name):
    """Draw 400K reference lines from the y-axis and separated value labels."""
    guide_start = steps[0] - 0.18
    guide_end = steps[-1]
    label_x = 1.2

    for method in method_order:
        final_value = data[model_name][method][-1]
        ax.hlines(
            final_value, guide_start, guide_end,
            color=colors[method], linestyle=(0, (4, 3)), linewidth=1.6,
            alpha=0.72, zorder=1,
        )
        ax.text(
            label_x,
            final_value + final_label_offsets[model_name][method],
            f'{final_value:.2f}',
            color=colors[method], fontsize=13, fontweight='bold',
            ha='left', va='center', zorder=4,
        )


def draw_curves(ax, model_name, show_labels=True):
    """Draw the three method curves on one axes."""
    model_data = data[model_name]
    for method in method_order:
        points = finite_points(model_data[method])
        if not points:
            continue
        x_values, y_values = zip(*points)
        is_ours = method == 'REPI + REPA'
        ax.plot(
            x_values, y_values,
            color=colors[method], linestyle=line_styles[method], marker=markers[method],
            linewidth=7.2 if is_ours else 6.0,
            markersize=11 if is_ours else 10,
            markerfacecolor='white',
            markeredgewidth=2.5 if is_ours else 2.0,
            alpha=0.88 if is_ours else 0.68,
            label=method if show_labels else '_nolegend_',
            zorder=3 if is_ours else 2,
        )


def format_legend(ax):
    legend = ax.legend(
        loc='upper right', frameon=True, fontsize=15,
        labelspacing=0.28, borderpad=0.42,
    )
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.95)
    for text, method in zip(legend.get_texts(), method_order):
        text.set_color(colors[method])
        text.set_fontweight('bold')
    return legend


def configure_independent_y_axis(ax, model_name):
    """Configure a separate linear range and regular ticks for each panel."""
    settings = panel_yaxis[model_name]
    ax.set_yscale('linear')
    ax.set_ylim(*settings['ylim'])
    ax.set_yticks(settings['ticks'])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:g}'))
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def draw_panel(ax, model_name):
    model_data = data[model_name]
    draw_final_guides(ax, model_name)
    draw_curves(ax, model_name)

    configure_independent_y_axis(ax, model_name)
    ax.set_xlim(steps[0] - 0.18, steps[-1] + 0.18)
    ax.set_xticks(steps)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Training Steps', labelpad=8)
    ax.set_ylabel(r'FID Score $\downarrow$', labelpad=8)
    ax.set_title(model_name, pad=11)
    ax.grid(True, linestyle='--', linewidth=1.0, color='#c9ccd1', alpha=0.10)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.45)
    ax.spines['bottom'].set_linewidth(1.45)
    ax.tick_params(axis='both', which='major', length=6, width=1.25)

    format_legend(ax)


output_dir = Path(__file__).resolve().parent

# 单独输出每个 panel，方便后续针对各自 y 轴继续微调。
for model_name, tag in zip(panel_order, panel_tags):
    fig, ax = plt.subplots(figsize=(5.1, 4.4), layout='constrained')
    draw_panel(ax, model_name)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'fid_chart_{tag}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

# 组合成 1 × 4 横排四 panel 图；各 ax 没有 sharey，因此 y 轴仍完全独立。
fig = plt.figure(figsize=(26.0, 5.9), layout='constrained')
subfigures = fig.subfigures(1, 4, wspace=0.035)
for subfig, model_name in zip(subfigures, panel_order):
    ax = subfig.subplots()
    draw_panel(ax, model_name)

for ext in ('pdf', 'png'):
    fig.savefig(output_dir / f'fid_chart_4panel.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

print('Done! Saved four individual panels and fid_chart_4panel.pdf/png.')
