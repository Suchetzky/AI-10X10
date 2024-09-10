import random
import csv


class Heuristics:
    holes_weight = -10
    empty_cells_weight = 10
    smoothness_weight = 7
    monotonicity_weight = 0
    merges_weight = 0
    sum_close_coordinates_values_weight = 0
    count_valid_moves_weight = 0
    weights = [holes_weight, empty_cells_weight, smoothness_weight, monotonicity_weight, merges_weight, sum_close_coordinates_values_weight, count_valid_moves_weight]
    @classmethod
    def write_weights_to_csv(cls, weights, heuristic_value):
        with open('data.csv', 'a') as csvfile:
            csvfile.write(f"{weights[0]},{weights[1]},{weights[2]},{weights[3]},{heuristic_value}\n")

    @staticmethod
    def holes(board):
        # Calculate the number of holes in the board
        holes = 0
        for col in range(board.width):
            for row in range(board.height):
                # If the cell is empty and the cells around are full
                if (board.grid[row][col] == 0 and
                        (row + 1 >= board.width or board.grid[row + 1][
                            col] == 1) and
                        (row - 1 <= 0 or board.grid[row - 1][col] == 1) and
                        (col - 1 <= 0 or board.grid[row][col - 1] == 1) and
                        (col + 1 >= board.height or board.grid[row][
                            col + 1] == 1)):
                    holes += 1
        return holes

    @staticmethod
    def bumpiness_cols(board):
        # Calculate the bumpiness of the board
        bumpiness = 0
        for col in range(board.width - 1):
            bumpiness += abs(sum([board.grid[row][col] for row in
                                  range(board.height)]) - sum(
                [board.grid[row][col + 1] for row in range(board.height)]))
        return bumpiness

    @staticmethod
    def bumpiness_rows(board):
        # Calculate rows bumpiness
        bumpiness = 0
        for row in range(board.height):
            bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row]))
        return bumpiness

    @staticmethod
    def empty_cells(board):
        # Calculate the number of empty cells
        empty_cells = 0
        for row in range(board.height):
            for col in range(board.width):
                if board.grid[row][col] == 0:
                    empty_cells += 1
        return empty_cells

    @staticmethod
    def calculate_smoothness(board):
        smoothness = 0
        for i in range(board.height):
            for j in range(board.height):
                if i + 1 < board.height:  # Compare vertically
                    smoothness -= abs(board.grid[i][j] - board.grid[i + 1][j])
                if j + 1 < len(board.grid[i]):  # Compare horizontally
                    smoothness -= abs(board.grid[i][j] - board.grid[i][j + 1])
        return smoothness

    @staticmethod
    def calculate_monotonicity(board):
        monotonicity = 0
        for i in range(board.height):
            row = board.grid[i]
            if row == sorted(row) or row == sorted(row, reverse=True):
                monotonicity += 1  # Row is monotonic
            col = [board.grid[j][i] for j in range(board.height)]
            if col == sorted(col) or col == sorted(col, reverse=True):
                monotonicity += 1  # Column is monotonic
        return monotonicity

    @staticmethod
    def count_merge_opportunities(board):
        merges = 0
        for i in range(board.height):
            for j in range(len(board.grid[i])):
                if i + 1 < board.height and board.grid[i][j] == board.grid[i + 1][j]:
                    merges += 1
                if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
                    merges += 1
        return merges

    @staticmethod
    def sum_close_coordinates_values(board):
        sum = 0
        for row in range(board.height):
            for col in range(board.width):
                if row + 1 < board.height:
                    sum += board.grid[row + 1][col]
                else:
                    sum += 1
                if row - 1 >= 0:
                    sum += board.grid[row - 1][col]
                else:
                    sum += 1
                if col + 1 < board.width:
                    sum += board.grid[row][col + 1]
                else:
                    sum += 1
                if col - 1 >= 0:
                    sum += board.grid[row][col - 1]
                else:
                    sum += 1
        return sum

    @staticmethod
    def count_valid_moves(board):
        valid_moves = 0
        for i in range(len(board.grid)):
            for j in range(len(board.grid[i])):
                if i + 1 < len(board.grid) and board.grid[i][j] == board.grid[i + 1][j]:
                    valid_moves += 1
                if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
                    valid_moves += 1
        return valid_moves

    @staticmethod
    def heuristic(board):
        # add weights to the different heuristics        
        return (Heuristics.holes_weight * Heuristics.holes(board) +
                        Heuristics.empty_cells_weight * Heuristics.empty_cells(board) +
                        Heuristics.smoothness_weight * Heuristics.calculate_smoothness(board) +
                        Heuristics.monotonicity_weight * Heuristics.calculate_monotonicity(board) +
                        Heuristics.merges_weight * Heuristics.count_merge_opportunities(board) +
                        Heuristics.sum_close_coordinates_values_weight * Heuristics.sum_close_coordinates_values(board) +
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
