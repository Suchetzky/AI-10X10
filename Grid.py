from tkinter import Canvas

# Define colors
black = 'black'
white = 'white'
blue = 'blue'


class Grid:
    def __init__(self, width, height, size):
        self.width = width
        self.height = height
        self.size = size
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def draw(self, canvas: Canvas):
        for row in range(self.height):
            for col in range(self.width):
                color = white if self.grid[row][col] == 0 else blue
                x1 = col * self.size
                y1 = row * self.size
                x2 = x1 + self.size
                y2 = y1 + self.size
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=black)

    def place_shape(self, shape, x, y):
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col] == 1:
                    self.grid[y + row][x + col] = 1

    # return all available index for a shape in the board just border check not the actual placement
    # if check_placement is True, it will also check if the shape can be placed in the board todo: check
    # shape is a list of lists not Shape!
    def can_place(self, shape, x, y, check_placement=False):
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col] == 1:
                    if (x + col >= self.width or y + row >= self.height or x + col < 0 or y + row < 0):
                        return False
                    if check_placement and self.grid[y + row][x + col] == 1:
                        return False
        return True

    def clear_lines(self):
        rows_to_clear = []
        columns_to_clear = []

        # save all rows that need to be cleared
        for row in range(self.height):
            if all(self.grid[row]):
                rows_to_clear.append(row)

        # save all columns that need to be cleared
        for col in range(self.width):
            if all([self.grid[row][col] for row in range(self.height)]):
                columns_to_clear.append(col)

        # clear all rows and columns
        for row in rows_to_clear:
            self.clear_single_line(row)

        for col in columns_to_clear:
            self.clear_single_column(col)

        return self.width * (len(rows_to_clear) + len(columns_to_clear))

    def clear_single_column(self, col):
        for i in range(self.width):
            self.grid[i][col] = 0

    def clear_single_line(self, row):
        for i in range(self.height):
            self.grid[row][i] = 0
