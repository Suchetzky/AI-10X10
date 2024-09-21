from tkinter import Canvas
import random

# Define colors for shapes
red = 'red'
blue = 'blue'


class Shape:
    # Predefined list of shape templates
    shapes = [
        # Shape 0: 2x2 square
        [[1, 1], [1, 1]],  # Original
        # * *
        # * *

        # Shape 1: 3x3 filled square
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Original
        # * * *
        # * * *
        # * * *

        # Shape 2: Single block
        [[1]],  # Original
        # *

        # Shape 3: 1x2 horizontal line
        [[1, 1]],  # Original
        # * *

        # Shape 4: 2x1 vertical line
        [[1], [1]],  # 90 degrees
        # *
        # *

        # Shape 5: 1x3 horizontal line
        [[1, 1, 1]],  # Original
        # * * *

        # Shape 6: 3x1 vertical line
        [[1], [1], [1]],  # 90 degrees
        # *
        # *
        # *

        #########################################

        # Shape 7: L-shape (3x3)
        [[1, 0, 0], [1, 0, 0], [1, 1, 1]],  # Original
        # *
        # *
        # * * *

        # Shape 8: Reverse L-shape (3x3)
        [[1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 90 degrees
        # * * *
        #     *
        #     *

        # Shape 9: Mirror L-shape (3x3)
        [[0, 0, 1], [0, 0, 1], [1, 1, 1]],  # 180 degrees
        #     *
        #     *
        # * * *

        # Shape 10: Inverted L-shape (3x3)
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],  # 270 degrees
        # * * *
        # *
        # *

        # Shape 11: 1x4 horizontal line
        [[1, 1, 1, 1]],  # Original
        # * * * *

        # Shape 12: 4x1 vertical line
        [[1], [1], [1], [1]],  # 90 degrees
        # *
        # *
        # *
        # *

        # Shape 13: Small L-shape (2x2)
        [[1, 0], [1, 1]],  # Original
        # *
        # * *

        # Shape 14: Reverse small L-shape (2x2)
        [[1, 1], [1, 0]],  # 90 degrees
        # * *
        # *

        # Shape 15: Inverted small L-shape (2x2)
        [[1, 1], [0, 1]],  # 180 degrees
        # * *
        #   *

        # Shape 16: Small S-shape (2x2)
        [[0, 1], [1, 1]],  # 270 degrees
        #   *
        # * *

        # Shape 17: 1x5 horizontal line
        [[1, 1, 1, 1, 1]],  # Original
        # * * * * *

        # Shape 18: 5x1 vertical line
        [[1], [1], [1], [1], [1]],  # 90 degrees
        # *
        # *
        # *
        # *
        # *
    ]

    def __init__(self, specific_shape_num=None):
        """
        Initialize a Shape object. If a specific shape number is provided,
        the corresponding shape is chosen, otherwise a random shape is selected.
        :param specific_shape_num: Optional, the index of a specific shape.
        """
        if specific_shape_num is not None:
            self.shape = self.shapes[specific_shape_num]
        else:
            self.shape = random.choice(self.shapes)

    def draw(self, canvas: Canvas, x, y, size):
        """
        Draw the shape on the provided canvas at a specific location and size.
        Each part of the shape is drawn as a filled rectangle.
        :param canvas: The canvas to draw on.
        :param x: The x coordinate (column) to start drawing.
        :param y: The y coordinate (row) to start drawing.
        :param size: The size of each block in the shape.
        """
        for row in range(len(self.shape)):
            for col in range(len(self.shape[0])):
                if self.shape[row][col] == 1:  # Only draw if the block is filled (marked as 1)
                    x1 = (x + col) * size
                    y1 = (y + row) * size
                    x2 = x1 + size
                    y2 = y1 + size
                    canvas.create_rectangle(x1, y1, x2, y2, fill=red, outline=blue)

    def rotate(self):
        """
        Rotate the shape 90 degrees clockwise by transposing and reversing rows.
        """
        self.shape = list(zip(*self.shape[::-1]))

    def get_part_size(self):
        """
        Calculate the total number of filled parts (1's) in the shape.
        :return: The total count of filled parts in the shape.
        """
        return sum(sum(row) for row in self.shape)

    def get_shape(self):
        """
        Get the current shape matrix.
        :return: The shape matrix (list of lists).
        """
        return self.shape

    def get_shape_num(self):
        """
        Get the index of the current shape from the predefined shapes list.
        :return: The index of the current shape.
        """
        return self.shapes.index(self.shape)

    def __eq__(self, other):
        """
        Compare if two Shape objects are equal by comparing their shape matrices.
        :param other: Another Shape object to compare with.
        :return: True if the shapes are identical, otherwise False.
        """
        return self.shape == other.shape
