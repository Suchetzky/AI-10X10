from tkinter import Canvas
import random

# Define colors
red = 'red'
blue = 'blue'


class Shape:
    shapes = [
        # Shape 0
        [[1, 1], [1, 1]],  # Original

        # * *
        # * *

        # Shape 1
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Original

        # * * *
        # * * *
        # * * *

        # Shape 2
        [[1]],  # Original

        # *

        # Shape 3
        [[1, 1]],  # Original

        # * *

        # shape 4
        [[1], [1]],  # 90 degrees

        # *
        # *


        # Shape 5
        [[1, 1, 1]],  # Original

        # * * *

        # Shape 6
        [[1], [1], [1]],  # 90 degrees

        # *
        # *
        # *

        #########################################

        # not a shape in the game
        #
        # [[1, 1], [0, 1], [0, 1]],  # 180 degrees
        #
        # # * *
        # #   *
        # #   *
        #
        # [[0, 0, 1], [1, 1, 1]],  # 270 degrees
        # # draw the shape with stars
        #
        # #     *
        # # * * *
        #
        # [[1, 0], [1, 0], [1, 1]],  # 270 degrees
        # draw the shape with stars

        # *
        # *
        # * *

        ###########################################

        # Shape 7
        [[1, 0, 0], [1, 0, 0], [1, 1, 1]],  # Original

        # *
        # *
        # * * *

        # Shape 8
        [[1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 90 degrees

        # * * *
        #     *
        #     *

        # Shape 9
        [[0, 0, 1], [0, 0, 1], [1, 1, 1]],  # 180 degrees

        #     *
        #     *
        # * * *

        # Shape 10
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],  # 270 degrees

        # * * *
        # *
        # *

        # Shape 11
        [[1, 1, 1, 1]],  # Original

        # * * * *

        # Shape 12
        [[1], [1], [1], [1]],  # 90 degrees

        # *
        # *
        # *
        # *

        # Shape 13
        [[1, 0], [1, 1]],  # Original

        # *
        # * *

        # Shape 14
        [[1, 1], [1, 0]],  # 90 degrees

        # * *
        # *

        # Shape 15
        [[1, 1], [0, 1]],  # 180 degrees

        # * *
        #   *

        # Shape 16
        [[0, 1], [1, 1]],  # 270 degrees

        #   *
        # * *

        # Shape 17
        [[1, 1, 1, 1, 1]],  # Original

        # * * * * *

        # Shape 18
        [[1], [1], [1], [1], [1]],  # 90 degrees

        # *
        # *
        # *
        # *
        # *

    ]

    def __init__(self, specific_shape_num=None):
        if specific_shape_num:
            self.shape = self.shapes[specific_shape_num]
        else:
            self.shape = random.choice(self.shapes)

    def draw(self, canvas: Canvas, x, y, size):
        """
        Draw the shape on the canvas
        :param canvas: canvas to draw on
        :param x: x coordinate of the shape
        :param y: y coordinate of the shape
        :param size: size of the shape
        :return:
        """
        for row in range(len(self.shape)):
            for col in range(len(self.shape[0])):
                if self.shape[row][col] == 1:
                    x1 = (x + col) * size
                    y1 = (y + row) * size
                    x2 = x1 + size
                    y2 = y1 + size
                    canvas.create_rectangle(x1, y1, x2, y2, fill=red, outline=blue)

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

    def get_part_size(self):
        return sum(sum(row) for row in self.shape)

    def get_shape(self):
        return self.shape

    def get_shape_num(self):
        return self.shapes.index(self.shape)

    # equal is the same shape
    def __eq__(self, other):
        return self.shape == other.shape
