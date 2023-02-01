import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
import pandas as pd
from sklearn import preprocessing

from features import wl, overlapping_segmentation, rms
from filters import process_signal

MODEL_PATH = "S1-13_G3_r_1674823233-49-0.825.model"
FILEPATH = "recordings/paper_1.csv"

n_samples = 52
skip = 5

ax2_fn = rms
ax3_fn = wl

signal_range = range(0, 8)
signal_range_length = len(signal_range)

# y = pd.read_csv("recordings/palm_open_0.csv", usecols = range(1, 9))
# y = pd.read_csv("michidk_dataset/s6_r_4/s6_r_4-paper-9-emg.csv", usecols = range(2, 10))
y = pd.read_csv(FILEPATH, usecols=signal_range)
x = range(1, y.shape[0] + 1)
y = np.array(y)

# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a
#
# def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = filtfilt(b, a, data)
#     return y

props = dict(boxstyle='round', facecolor='white', alpha=0.35)
default_alpha_val = 0.2

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(8, 6))

ax1.set_title('Press left mouse button and drag '
              'to select a region in the top graph')

import colorsys

HSV_tuples = [(x * 1.0 / signal_range_length, 0.5, 0.5) for x in range(signal_range_length)]
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

    # new_plot = ax1.plot(x, col, c=RGB_tuples[idx], alpha=default_alpha_val)
    # signal_plots.append(new_plot)

    fs = 200
    cutoff = fs / 4
    order = 1

    ## col_filtered = butter_lowpass_filtfilt(col, cutoff=cutoff, fs=fs, order=5)  # cutoff: [0, fs)
    emg_filtered, emg_rectified, emg_envelope = process_signal(col, order=4, low_pass=10, sfreq=200, high_band=4, low_band=45)
    new_plot_filtered = ax1.plot(x, emg_rectified, c='red', alpha=default_alpha_val)
    signal_plots.append(new_plot_filtered)

    ## col_filtered = butter_lowpass_filtfilt(col, cutoff=cutoff, fs=fs*2, order=5)  # cutoff: [0, fs)
    emg_filtered, emg_rectified, emg_envelope = process_signal(col, order=5, low_pass=10, sfreq=200, high_band=4, low_band=90)
    new_plot_filtered = ax1.plot(x, emg_rectified, c='blue', alpha=default_alpha_val)
    signal_plots.append(new_plot_filtered)


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


hover_cid = None
leave_axes_cid = None


def on_selector_move(xmin, xmax):
    global hover_cid;
    global leave_axes_cid
    if hover_cid and leave_axes_cid:
        reset()
        fig.canvas.mpl_disconnect(hover_cid)
        fig.canvas.mpl_disconnect(leave_axes_cid)
        hover_cid = None
        leave_axes_cid = None


desired_range = 200

selected_data = None

is_calculate_features = False


def on_selector_select(xmin, xmax):
    xmin = int(xmin)
    xmax = int(xmax)
    global selected_data
    x_range = x[-1] - x[0] + 1

    if x_range < desired_range:
        print(f"Data ({x_range}) is not long enough to hold the desired range ({desired_range})")

    min_out_of_bounds = xmin < 0
    max_out_of_bounds = xmax > x_range
    selection_range = xmax - xmin
    is_short_length = selection_range != desired_range
    new_xmin = xmin
    new_xmax = xmax

    if min_out_of_bounds or max_out_of_bounds or is_short_length:
        if min_out_of_bounds:
            new_xmin = max(0, xmin)
            new_xmax = new_xmin + desired_range

        if max_out_of_bounds:
            new_xmax = min(x[-1], xmax)
            new_xmin = new_xmax - desired_range

        if is_short_length:
            new_xmax = new_xmin + desired_range

        span.extents = (new_xmin, new_xmax)
        span.onselect(new_xmin, new_xmax)
        return
    xmin = new_xmin
    xmax = new_xmax

    global hover_cid;
    global leave_axes_cid
    hover_cid = fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    leave_axes_cid = fig.canvas.mpl_connect('axes_leave_event', leave_axes)

    global feature_plots_1;
    global feature_plots_2;
    global last_highlighted_signal_index
    last_highlighted_signal_index = -1

    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    # indmax = min(desired_range, indmax)

    region_x = x[indmin:indmax]

    ax2.clear()
    ax3.clear()

    selected_data = y[indmin:indmax]

    if is_calculate_features:
        signals_length = len(y[indmin:indmax].T)
        feature_plots_1 = [[] for i in range(signals_length)]
        feature_plots_2 = [[] for i in range(signals_length)]
        ax_plot_pairs[1] = (ax2, feature_plots_1)
        ax_plot_pairs[2] = (ax3, feature_plots_2)

        for idx, col in enumerate(y[indmin:indmax].T):
            region_y = col  # for each channel
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
                        perc = (segment_id / len(segments))
                        ax2_plot = ax2.plot(segment_x, [ax2_val] * segment_x_length, c=RGB_tuples[idx], linestyle='-',
                                            linewidth=3.0, alpha=1)
                        ax3_plot = ax3.plot(segment_x, [ax3_val] * segment_x_length, c=RGB_tuples[idx], linestyle='-',
                                            linewidth=3.0, alpha=1)

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
    set_alpha(feature_plots_1, 0.2)
    set_alpha(feature_plots_2, 0.2)
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
    # minspan=200,
    span_stays=True,
    onmove_callback=on_selector_move,
    grab_range=1,
    useblit=True,
    props=dict(alpha=0.2, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)
s = 150
span.extents = (s, s+200)
span.onselect(s, s+200)

predictButton = Button(ax4, 'Predict')

from rnn import model_test

model = None


def predict(e):
    global selected_data;
    global model
    if model is None:
        model = model_test.load_best_model()

    if selected_data is not None:
        gestures = ['rock', 'paper', 'scissors']  # , 'paper', 'scissors'
        lb = preprocessing.LabelBinarizer()

        selected_data = selected_data.reshape(-1, 8, 200)
        selected_data_norm = np.interp(selected_data, (-128, 127), (-1, +1))
        n_steps, n_length, n_features = 4, 50, 8  # ops happen on 50
        selected_data_norm = selected_data_norm.reshape(1, 200, 8)
        x_data = selected_data_norm.reshape((selected_data_norm.shape[0], n_steps, n_length, n_features))

        # print(x_data)
        prediction = model.predict(x_data)
        print(prediction)
        lb.fit(gestures)
        pred_dec = lb.inverse_transform(prediction)
        all = lb.transform(gestures)
        print(all, pred_dec)


predictButton.on_clicked(predict)

# Set useblit=True on most backends for enhanced performance.


plt.show()
