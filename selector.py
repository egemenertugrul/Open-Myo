import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd

from features import disjoint_segmentation, wl, overlapping_segmentation, rms, mav, zc
from matplotlib import collections  as mc

n_samples = 52
skip = 5

props = dict(boxstyle='round', facecolor='white', alpha=0.35)
default_alpha_val = 0.2

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 6))

ax1.set_title('Press left mouse button and drag '
              'to select a region in the top graph')

signal_range = range(0, 8)
signal_range_length = len(signal_range)

y = pd.read_csv("recordings/palm_open_0.csv", usecols = signal_range)
x = range(1, y.shape[0] + 1)
y = np.array(y)

ax2_fn = rms
ax3_fn = wl

import colorsys
HSV_tuples = [(x*1.0/signal_range_length, 0.5, 0.5) for x in range(signal_range_length)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

signal_plots = []
feature_plots_1 = []
feature_plots_2 = []

# ax_plot_dict = {ax1: signal_plots, ax2: feature_plots_1, ax3: feature_plots_2}
ax_plot_pairs = [(ax1, signal_plots), (ax2, feature_plots_1), (ax3, feature_plots_2)]


for idx, col in enumerate(y.T):
    # col = col.squeeze(axis=1)
    col = np.interp(col, (-128, 127), (-1, +1))
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-1, 1)
    new_plot = ax1.plot(x, col, c=RGB_tuples[idx], alpha=default_alpha_val)
    signal_plots.append(new_plot)


# lc = mc.LineCollection(lines, colors=c, linewidths=2)

# line2, = ax2.plot([], [])

def set_alpha(plots, alpha):
    if not isinstance(plots, np.ndarray):
        plots = np.array(plots)

    plots = plots.flatten()
    for plot in plots:
        plot.set_alpha(alpha)

    fig.canvas.draw_idle()

def highlight_single(all_curves_in_subplot, signal_id):
    if len(all_curves_in_subplot) == 0:
        return

    set_alpha(all_curves_in_subplot, default_alpha_val)
    set_alpha(all_curves_in_subplot[signal_id], 1)


hover_cid = -1
leave_axes_cid = -1

def on_selector_move(xmin, xmax):
    reset()
    global hover_cid; global leave_axes_cid
    fig.canvas.mpl_disconnect(hover_cid)
    fig.canvas.mpl_disconnect(leave_axes_cid)

def on_selector_select(xmin, xmax):
    global hover_cid; global leave_axes_cid
    hover_cid = fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    leave_axes_cid = fig.canvas.mpl_connect('axes_leave_event', leave_axes)


    global feature_plots_1; global feature_plots_2; global last_highlighted_signal_index
    last_highlighted_signal_index = -1

    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax)

    region_x = x[indmin:indmax]

    ax2.clear()
    ax3.clear()

    signals_length = len(y[indmin:indmax].T)
    feature_plots_1 = [[] for i in range(signals_length)]
    feature_plots_2 = [[] for i in range(signals_length)]
    ax_plot_pairs[1] = (ax2, feature_plots_1)
    ax_plot_pairs[2] = (ax3, feature_plots_2)

    for idx, col in enumerate(y[indmin:indmax].T):
        region_y = col
        # region_y = y[indmin:indmax]

        if len(region_x) >= 2:
            segments = overlapping_segmentation((region_x, region_y), n_samples=n_samples, skip=skip)

            if segments is not None:
                ax2_3_x_data = []
                ax2_y_data = []
                ax3_y_data = []

                for segment_id, segment in enumerate(segments):
                    (segment_x, segment_y) = segment
                    segment_x_length = len(segment_x)
                    ax2_val = ax2_fn(segment_y)
                    ax3_val = ax3_fn(segment_y)

                    ax2_3_x_data.extend(segment_x)
                    ax2_y_data.extend([ax2_val] * segment_x_length)
                    ax3_y_data.extend([ax3_val] * segment_x_length)
                    perc = (segment_id/len(segments))
                    ax2_plot = ax2.plot(segment_x, [ax2_val] * segment_x_length, c=RGB_tuples[idx], linestyle='-', linewidth=3.0, alpha=1)
                    ax3_plot = ax3.plot(segment_x, [ax3_val] * segment_x_length, c=RGB_tuples[idx], linestyle='-', linewidth=3.0, alpha=1)

                    feature_plots_1[idx].append(ax2_plot)
                    feature_plots_2[idx].append(ax3_plot)

                ax2_3_x_data = np.array(ax2_3_x_data)
                ax2_y_data = np.array(ax2_y_data)
                ax3_y_data = np.array(ax3_y_data)
                # line2.set_data(x_data, y_data)

                ax2.set_xlim(x[0], x[-1])
                ax3.set_xlim(x[0], x[-1])
                # ax2.set_xlim(ax2_3_x_data[0], ax2_3_x_data[-1])
                # ax3.set_xlim(ax2_3_x_data[0], ax2_3_x_data[-1])
                # ax2.set_ylim(0, ax2_y_data.max() * 1.20)
                # ax3.set_ylim(0, ax3_y_data.max() * 1.20)
            else:
                # line2.set_data([], [])
                ax2.plot([], [])
                ax3.plot([], [])
            # line2.set_data(region_x, region_y)
            # ax2.set_xlim(region_x[0], region_x[-1])
            # ax2.set_ylim(region_y.min(), region_y.max())
            fig.canvas.draw_idle()

# annot = ax2.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                     bbox=dict(boxstyle="round", fc="w"),
#                     arrowprops=dict(arrowstyle="->"))
# annot.set_visible(False)

signal_index_text = None
last_highlighted_signal_index = -1

def update_text(signal_index):
    global signal_index_text
    if signal_index_text:
        signal_index_text.remove()
        signal_index_text = None

    if signal_index >= 0:
        signal_index_text = ax1.text(0.05, 0.95, str(signal_index + 1), transform=ax1.transAxes, fontsize=12,
                                 verticalalignment='top', bbox=props)

def get_ids_from_segment_curve(plots, curve):
    plots = np.array(plots)
    elem = np.array(np.where((plots == curve))).flatten()
    if elem.size == 0:
        return -1, -1

    signal_index = elem[0]
    segment_index = elem[1]
    return signal_index, segment_index

def on_plot_hover(event):
    for pair in ax_plot_pairs:
        ax, plot = pair
        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                signal_index, segment_index = get_ids_from_segment_curve(plot, curve)

                if signal_index >= 0 and segment_index >= 0:
                    update_all_plots(signal_index)

def update_all_plots(signal_index):
    global last_highlighted_signal_index
    if last_highlighted_signal_index != signal_index:
        last_highlighted_signal_index = signal_index

        highlight_single(signal_plots, signal_index)
        update_text(signal_index)

        highlight_single(feature_plots_1, signal_index)
        highlight_single(feature_plots_2, signal_index)

def reset():
    set_alpha(signal_plots, 0.2)
    set_alpha(feature_plots_1, 1)
    set_alpha(feature_plots_2, 1)
    update_text(-1)

def leave_axes(event):
    for pair in ax_plot_pairs:
        ax, plot = pair
        if ax.contains(event)[0]:
            reset()

span = SpanSelector(
    ax1,
    on_selector_select,
    "horizontal",
    onmove_callback=on_selector_move,
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)
# Set useblit=True on most backends for enhanced performance.


plt.show()