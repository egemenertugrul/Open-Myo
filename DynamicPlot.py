import math

from matplotlib import pyplot as plt

plt.ion()


class AxesWrapper:
    def __init__(self, lines, ax, max_display_capacity=None):
        self.lines = lines
        self.ax = ax
        self.ax.set_autoscalex_on(True)
        self.ax.set_autoscaley_on(True)
        self.ax.grid()
        self.x_data = []
        self.y_data = []
        self.max_display_capacity = max_display_capacity

    def add(self, data):
        x_data, y_data = data
        self.x_data.append(x_data)
        self.y_data.append(y_data)

        if self.max_display_capacity:
            if len(self.x_data) > self.max_display_capacity:
                self.x_data.pop(0)

            if len(self.y_data) > self.max_display_capacity:
                self.y_data.pop(0)

        # self.lines.set_xdata(self.x_data)
        # self.lines.set_ydata(self.y_data)
        self.lines.set_data(self.x_data, self.y_data)

    def __getattr__(self, item):
        return getattr(self.ax, item)

    def flush(self):
        self.ax.relim()
        self.ax.autoscale_view()


class Plot:
    def set_title(self, rowcol, title):
        row, col = rowcol
        row = row - 1
        col = col - 1
        self.axes[row][col].set_title(title)

    def add_to(self, rowcol, xy_val):
        row, col = rowcol
        row = row - 1
        col = col - 1
        if isinstance(xy_val, list):
            for v in xy_val:
                self.axes[row][col].add(v)
        else:
            self.axes[row][col].add(xy_val)

    def flush(self):
        for i in range(len(self.axes)):
            for j in range(len(self.axes[0])):
                self.axes[i][j].flush()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __init__(self, rowcol, max_display_capacity=None, style_args=None):
        rows, cols = rowcol
        self.figure, axes = plt.subplots(rows, cols, squeeze=False, sharex='none', sharey='none')
        self.figure.tight_layout()

        row_count = len(axes)
        col_count = len(axes[0])

        self.axes = []
        self.lines = []
        for i in range(row_count):
            self.axes.append([])
            for j in range(col_count):
                ax = axes[i, j]

                if style_args:
                    style = style_args[i][j]
                    lines, = ax.plot([], [], *style)
                else:
                    lines, = ax.plot([], [])


                self.lines.append(lines)
                ax.set_title(str(i + j))

                new_ax = AxesWrapper(lines, ax, max_display_capacity)
                self.axes[i].append(new_ax)


if __name__ == '__main__':

    d_plot = Plot(rowcol=(3, 3),
                  max_display_capacity=10,
                  style_args=[
        [
            ['r'], ['g'], ['b']
        ], [
            ['r-'], ['g-'], ['b-']
        ], [
            ['ro'], ['go'], ['bo']
        ]
    ])

    # Set titles
    d_plot.set_title((1, 1), "1")
    d_plot.set_title((1, 2), "2")
    d_plot.set_title((1, 3), "3")

    d_plot.set_title((2, 1), "1")
    d_plot.set_title((2, 2), "2")
    d_plot.set_title((2, 3), "3")

    d_plot.set_title((3, 1), "4")
    d_plot.set_title((3, 2), "5")
    d_plot.set_title((3, 3), "6")

    i = 0
    interval = 0.250

    while True:
        a = math.sin(i)
        b = math.cos(i)
        c = math.sin(math.cos(i))

        # Set data
        d_plot.add_to((1, 1), (i, a))
        d_plot.add_to((2, 1), (i, a ** 2))
        d_plot.add_to((3, 1), (i, a ** 3))

        d_plot.add_to((1, 2), (i, b))
        d_plot.add_to((2, 2), (i, b ** 2))
        d_plot.add_to((3, 2), (i, b ** 3))

        d_plot.add_to((1, 3), (i, c))
        d_plot.add_to((2, 3), (i, c ** 2))
        d_plot.add_to((3, 3), (i, c ** 3))

        i = i + interval

        d_plot.flush()
