import cairo
import numpy as np
import os

from dynamic_models.synthetic_dynamic_model_1 import SyntheticDynamicModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


ROW = 1  # pixels
COL = 50  # pixels
LINE_WIDTH = 0.01
COLORS = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
]


def _draw_matrix(matrix, output_path):
    width = COL * matrix.shape[1]
    height = ROW * matrix.shape[0]

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.scale(width, height)

    # matrix brackets
    ctx.move_to(0, 0)
    ctx.line_to(0, 1)
    ctx.move_to(0, 0)
    ctx.line_to(COL / 8 / width, 0)
    ctx.move_to(0, 1)
    ctx.line_to(COL / 8 / width, 1)
    ctx.move_to(1, 0)
    ctx.line_to(1, 1)
    ctx.move_to(1, 0)
    ctx.line_to(1 - COL / 8 / width, 0)
    ctx.move_to(1, 1)
    ctx.line_to(1 - COL / 8 / width, 1)
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_line_width(LINE_WIDTH)
    ctx.stroke()

    # matrix contents
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        column_min = np.min(column)
        column_max = np.max(column)
        for i in range(matrix.shape[0]):
            y = (ROW / 2 + i * ROW) / height
            x = (j * COL + COL / 8 + COL * 3 / 4 * (matrix[i, j] - column_min) / (column_max - column_min)) / width
            if i == 0:
                ctx.move_to(x, y)
            else:
                ctx.line_to(x, y)
        ctx.set_source_rgb(*COLORS[j % len(COLORS)])
        ctx.set_line_width(LINE_WIDTH)
        ctx.stroke()

    surface.write_to_png(output_path)


def run():
    network = FullyConnectedRandomWeights(3)
    dynamic_model = SyntheticDynamicModel1(network)
    x = dynamic_model.get_x(300)
    y = dynamic_model.get_x_dot(x)
    _draw_matrix(x, os.path.join(OUTPUT_DIR, 'x.png'))
    _draw_matrix(y, os.path.join(OUTPUT_DIR, 'y.png'))


if __name__ == '__main__':
    run()
