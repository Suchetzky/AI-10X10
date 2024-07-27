import pygame
import random

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
class Grid:
    def __init__(self, width, height, size):
        self.width = width
        self.height = height
        self.size = size
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def draw(self, screen):
        for row in range(self.height):
            for col in range(self.width):
                color = white if self.grid[row][col] == 0 else blue
                pygame.draw.rect(screen, color, (col * self.size, row * self.size, self.size, self.size))
                pygame.draw.rect(screen, black, (col * self.size, row * self.size, self.size, self.size), 1)

    def place_shape(self, shape, x, y):
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col] == 1:
                    self.grid[y + row][x + col] = 1

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
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = self.height - len(new_grid)
        self.grid = [[0] * self.width for _ in range(lines_cleared)] + new_grid

        # Clear columns
        for col in range(self.width):
            if all(self.grid[row][col] == 1 for row in range(self.height)):
                for row in range(self.height):
                    self.grid[row][col] = 0
                lines_cleared += 1

        return lines_cleared * 10
