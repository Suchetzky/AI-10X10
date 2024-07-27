import pygame
import random

# Define colors
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)


class Shape:
    shapes = [
        [[1, 1], [1, 1]],  # Square
        [[1, 1]],  # Small line
        [[1, 1, 1, 1]],  # Line horizontal
        [[1, 1, 1], [0, 1, 0]],  # T shape
        [[1, 1, 0], [0, 1, 1]],  # Z shape
        [[0, 1, 1], [1, 1, 0]],  # S shape
        [[1, 1, 1], [1, 0, 0]],  # L shape
    ]

    def __init__(self):
        self.shape = random.choice(self.shapes)

    def draw(self, screen, x, y, size):
        for row in range(len(self.shape)):
            for col in range(len(self.shape[0])):
                if self.shape[row][col] == 1:
                    pygame.draw.rect(screen, red, ((x + col) * size, (y + row) * size, size, size))

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))
