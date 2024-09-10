import random
import csv
import numpy as np


class Heuristics:
    _instance = None

    holes_weight = random.randint(-10, 0)
    empty_cells_weight = random.randint(0, 10)
    smoothness_weight = random.randint(0, 10)
    monotonicity_weight = random.randint(0, 10)
    merges_weight = random.randint(0, 10)
    sum_close_coordinates_values_weight = random.randint(-10, 10)
    count_valid_moves_weight = random.randint(-10, 10)
    weights = [holes_weight, empty_cells_weight, smoothness_weight, monotonicity_weight, merges_weight, sum_close_coordinates_values_weight, count_valid_moves_weight]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Heuristics, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    @classmethod
    
    def random_weights(cls):
        cls.holes_weight = random.randint(-10, 10)
        cls.empty_cells_weight = random.randint(-10, 10)
        cls.smoothness_weight = random.randint(-10, 10)
        cls.monotonicity_weight = random.randint(-10, 10)
        cls.merges_weight = random.randint(-10, 10)
        # cls.sum_close_coordinates_values_weight = random.randint(-10, 10)
        cls.count_valid_moves_weight = random.randint(0, 10)
        cls.weights = [cls.holes_weight, cls.empty_cells_weight, cls.smoothness_weight, cls.monotonicity_weight, cls.merges_weight, cls.count_valid_moves_weight]
   
    @classmethod
    def write_weights_to_csv(cls, heuristic_value):
        with open('data.csv', 'a') as csvfile:
            for i in range(len(Heuristics.weights)):
                csvfile.write(f"{Heuristics.weights[i]},")
            csvfile.write(f"{heuristic_value}\n")
    @staticmethod
    def holes(board):
        # Calculate the number of holes in the board
        holes = 0
        for col in range(board.width):
            for row in range(board.height):
                # If the cell is empty and the cells around are full
                if (board.grid[row][col] == 0 and
                        (row + 1 >= board.width or board.grid[row + 1][col] == 1) and
                        (row - 1 <= 0 or board.grid[row - 1][col] == 1) and
                        (col - 1 <= 0 or board.grid[row][col - 1] == 1) and
                        (col + 1 >= board.height or board.grid[row][col + 1] == 1)):
                    holes += 1
        return holes

    @staticmethod
    def bumpiness_cols(board):
        # Calculate the bumpiness of the board
        # bumpiness = 0
        # for col in range(board.width - 1):
        #     bumpiness += abs(sum([board.grid[row][col] for row in
        #                           range(board.height)]) - sum(
        #         [board.grid[row][col + 1] for row in range(board.height)]))
        # return bumpiness
        col_sums = np.sum(board.grid, axis=0)
        return np.sum(np.abs(np.diff(col_sums)))

    @staticmethod
    def bumpiness_rows(board):
        # Calculate rows bumpiness
        # bumpiness = 0
        # for row in range(board.height):
        #     bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row]))
        # return bumpiness
        row_sums = np.sum(board.grid, axis=1)
        return np.sum(np.abs(np.diff(row_sums)))

    @staticmethod
    def empty_cells(board):
        # Calculate the number of empty cells
        # empty_cells = 0
        # for row in range(board.height):
        #     for col in range(board.width):
        #         if board.grid[row][col] == 0:
        #             empty_cells += 1
        # return empty_cells
        return np.sum(board.grid == 0)

    @staticmethod
    def calculate_smoothness(board):
        # smoothness = 0
        # for i in range(board.height):
        #     for j in range(board.height):
        #         if i + 1 < board.height:  # Compare vertically
        #             smoothness -= abs(board.grid[i][j] - board.grid[i + 1][j])
        #         if j + 1 < len(board.grid[i]):  # Compare horizontally
        #             smoothness -= abs(board.grid[i][j] - board.grid[i][j + 1])
        # return smoothness
        smoothness = 0
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=0)))  # Vertical smoothness
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=1)))  # Horizontal smoothness
        return smoothness

    @staticmethod
    def calculate_monotonicity(board):
        # monotonicity = 0
        # for i in range(board.height):
        #     row = board.grid[i]
        #     if row == sorted(row) or row == sorted(row, reverse=True):
        #         monotonicity += 1  # Row is monotonic
        #     col = [board.grid[j][i] for j in range(board.height)]
        #     if col == sorted(col) or col == sorted(col, reverse=True):
        #         monotonicity += 1  # Column is monotonic
        # return monotonicity
        grid = np.array(board.grid)
        # Vectorized smoothness calculation
        smoothness = 0
        smoothness -= np.sum(
            np.abs(np.diff(grid, axis=0)))  # Vertical smoothness
        smoothness -= np.sum(
            np.abs(np.diff(grid, axis=1)))  # Horizontal smoothness
        return smoothness

    @staticmethod
    def count_merge_opportunities(board):
        # merges = 0
        # for i in range(board.height):
        #     for j in range(len(board.grid[i])):
        #         if i + 1 < board.height and board.grid[i][j] == board.grid[i + 1][j]:
        #             merges += 1
        #         if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
        #             merges += 1
        # return merges
        grid = np.array(board.grid)
        merges = 0
        # Vectorized checking for adjacent merges (rows)
        merges += np.sum(grid[:, :-1] == grid[:, 1:])
        # Vectorized checking for adjacent merges (columns)
        merges += np.sum(grid[:-1, :] == grid[1:, :])
        return merges

    # @staticmethod
    # def sum_close_coordinates_values(board):
    #     adjacent_sum = 0
    #     for row in range(board.height):
    #         for col in range(board.width):
    #             if board.grid[row][col] != 0:
    #                 # Check up, down, left, right
    #                 if row > 0:
    #                     adjacent_sum += board.grid[row - 1][col]
    #                 if row < board.height - 1:
    #                     adjacent_sum += board.grid[row + 1][col]
    #                 if col > 0:
    #                     adjacent_sum += board.grid[row][col - 1]
    #                 if col < board.width - 1:
    #                     adjacent_sum += board.grid[row][col + 1]
    #     return adjacent_sum
        # sum = 0
        # for row in range(board.height):
        #     for col in range(board.width):
        #         if row + 1 < board.height:
        #             sum += board.grid[row + 1][col]
        #         else:
        #             sum += 1
        #         if row - 1 >= 0:
        #             sum += board.grid[row - 1][col]
        #         else:
        #             sum += 1
        #         if col + 1 < board.width:
        #             sum += board.grid[row][col + 1]
        #         else:
        #             sum += 1
        #         if col - 1 >= 0:
        #             sum += board.grid[row][col - 1]
        #         else:
        #             sum += 1
        # return sum

    @staticmethod
    def count_valid_moves(board):
        grid = np.array(board.grid)
        valid_moves = 0
        valid_moves += np.sum(grid[:, :-1] == grid[:, 1:])  # Horizontal
        valid_moves += np.sum(grid[:-1, :] == grid[1:, :])  # Vertical
        return valid_moves
        # valid_moves = 0
        # for i in range(len(board.grid)):
        #     for j in range(len(board.grid[i])):
        #         if i + 1 < len(board.grid) and board.grid[i][j] == board.grid[i + 1][j]:
        #             valid_moves += 1
        #         if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
        #             valid_moves += 1
        # return valid_moves

    @staticmethod
    def heuristic(board):
        # add weights to the different heuristics        
        return (Heuristics.holes_weight * Heuristics.holes(board) +
                        Heuristics.empty_cells_weight * Heuristics.empty_cells(board) +
                        Heuristics.smoothness_weight * Heuristics.calculate_smoothness(board) +
                        Heuristics.monotonicity_weight * Heuristics.calculate_monotonicity(board) +
                        Heuristics.merges_weight * Heuristics.count_merge_opportunities(board) +
                        # Heuristics.sum_close_coordinates_values_weight * Heuristics.sum_close_coordinates_values(board) +
                        Heuristics.count_valid_moves_weight * Heuristics.count_valid_moves(board))

# if __name__ == '__main__':
#     from Grid import Grid
# 
#     board = Grid(4, 4, 50)
#     holes_weight = random.randint(-10, 0)
#     bumpiness_cols_weight = random.randint(-10, 0)
#     bumpiness_rows_weight = random.randint(-10, 0)
#     blocks_of_shapes_weight = random.randint(0, 15)
# 
#     heuristic_value = Heuristics.heuristic(board,holes_weight, bumpiness_cols_weight, bumpiness_rows_weight, blocks_of_shapes_weight)
#     print(heuristic_value)
# 
#     with open('data.csv', 'a') as csvfile:
#         csvfile.write(f"{holes_weight},{bumpiness_cols_weight},{bumpiness_rows_weight},{blocks_of_shapes_weight},{heuristic_value}\n")
