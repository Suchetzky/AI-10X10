import random
import csv
from scipy.cluster.hierarchy import weighted
import numpy as np


class Heuristics:

    def holes(self, board):
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

    def bumpiness_cols(self, board):
        # Calculate the bumpiness of the board
        bumpiness = 0
        for col in range(board.width - 1):
            bumpiness += abs(sum([board.grid[row][col] for row in
                                  range(board.height)]) - sum(
                [board.grid[row][col + 1] for row in range(board.height)]))
        return bumpiness

    def bumpiness_rows(self, board):
        # Calculate rows bumpiness
        bumpiness = 0
        for row in range(board.height):
            bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row]))
        return bumpiness

    def empty_cells(self, board):
        grid = np.array(board.grid)
        return np.sum(grid == 0)

    def calculate_smoothness(self, board):
        smoothness = 0
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=0)))  # Vertical smoothness
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=1)))  # Horizontal smoothness
        return smoothness

    def calculate_monotonicity(self, board):
        grid = np.array(board.grid)
        monotonicity = 0
        for row in grid:
            if np.all(np.diff(row) >= 0) or np.all(np.diff(row) <= 0):
                monotonicity += 1
        for col in grid.T:  # Transpose for columns
            if np.all(np.diff(col) >= 0) or np.all(np.diff(col) <= 0):
                monotonicity += 1
        return monotonicity

    def count_merge_opportunities(self, board):
        merges = 0
        for i in range(board.height):
            for j in range(len(board.grid[i])):
                if i + 1 < board.height and board.grid[i][j] == board.grid[i + 1][j]:
                    merges += 1
                if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
                    merges += 1
        return merges


    def count_valid_moves(self, board):
        valid_moves = 0
        for i in range(len(board.grid)):
            for j in range(len(board.grid[i])):
                if i + 1 < len(board.grid) and board.grid[i][j] == board.grid[i + 1][j]:
                    valid_moves += 1
                if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
                    valid_moves += 1
        return valid_moves

    def corner_heuristic(self, board):
        grid = np.array(board.grid)
        corners = [
            grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]
        ]
        return sum(corner for corner in corners)

    def edge_heuristic(self, board):
        grid = np.array(board.grid)
        top_row = np.sum(grid[0, :])
        bottom_row = np.sum(grid[-1, :])
        left_col = np.sum(grid[:, 0])
        right_col = np.sum(grid[:, -1])
        return top_row + bottom_row + left_col + right_col

    def border_heuristic(self, board):
        # rewards placing tiles or blocks near the border
        score = 0
        grid = board.grid
        border = board.height
        for j in range(border):
            if grid[0][j] != 0:
                score += 1
            if grid[border - 1][j] != 0:
                score += 1
            if grid[j][0] != 0:
                score += 1
            if grid[j][border - 1] != 0:
                score += 1
        return score

    def heuristic1_adjacent_pairs(self, board):
        rows = len(board.grid)
        cols = len(board.grid[0])
        score = 0

        # Check horizontal adjacent pairs
        for i in range(rows):
            for j in range(cols - 1):
                if (board.grid[i][j] == board.grid[i][j + 1]):
                    score += 1

        # Check vertical adjacent pairs
        for i in range(rows - 1):
            for j in range(cols):
                if (board.grid[i][j] == board.grid[i + 1][j]):
                    score += 1

        return score

    def heuristic2_2x2_blocks(self, board):
        rows = len(board.grid)
        cols = len(board.grid[0])
        score = 0

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get the 2x2 block
                block = [board.grid[i][j], board.grid[i][j + 1], board.grid[i + 1][j],
                         board.grid[i + 1][j + 1]]
                occupied_count = block.count(
                    1)  # Assuming 1 represents occupied, 0 represents empty

                if occupied_count == 4 or occupied_count == 0:
                    # Best score: all 4 are either occupied or empty
                    score += 3
                elif (block[0] == block[1] and block[2] == block[3]) or (
                        block[0] == block[2] and block[1] == block[3]):
                    # Second best score: One row or one column is fully occupied or fully empty
                    score += 2
                elif occupied_count == 3 or occupied_count == 1:
                    # Third best score: Three cells are occupied and one is empty, or vice versa
                    score += 1
                elif occupied_count == 2:
                    # Worst score: chessboard-like pattern
                    if (block[0] != block[1]) and (block[2] != block[3]) and (
                            block[0] != block[2]):
                        score -= 1  # Penalty for chessboard-like patterns

        return score

    def can_place_block(self, board, start_row, start_col, height, width):
        # Check if a block of size (height, width) can fit starting at (start_row, start_col)
        for i in range(start_row, start_row + height):
            for j in range(start_col, start_col + width):
                if board.grid[i][j] != 0:  # Assuming 0 represents an empty cell
                    return False
        return True

    def large_shape_fit_heuristic(self, board):
        rows = len(board.grid)
        cols = len(board.grid[0])
        large_block_count = 0

        # Check for 3x3 block placement
        for i in range(rows - 2):
            for j in range(cols - 2):
                if self.can_place_block(board, i, j, 3, 3):
                    large_block_count += 1

        # Check for 5x1 block placement (horizontal)
        for i in range(rows):
            for j in range(cols - 4):
                if self.can_place_block(board, i, j, 1, 5):
                    large_block_count += 1

        # Check for 1x5 block placement (vertical)
        for i in range(rows - 4):
            for j in range(cols):
                if self.can_place_block(board, i, j, 5, 1):
                    large_block_count += 1

        return large_block_count

    def heuristic(self, board):
        count_valid_moves_weight = 0
        heur1_weight = 0
        heur2_weight = 1
        return (
                + heur2_weight * self.heuristic2_2x2_blocks(board)
                + heur1_weight * self.heuristic1_adjacent_pairs(board)
                + count_valid_moves_weight * self.count_valid_moves(board)
                )


if __name__ == '__main__':
    # Test the heuristics
    from Grid import Grid

    grid = Grid(4, 4, 50)
    heuristics = Heuristics()

    print(heuristics.heuristic(grid))
    with open('data.csv', 'a') as csvfile:
        csvfile.write("9000\n")

    print(heuristics.heuristic(grid))

    with open('data.csv', 'a') as csvfile:
        csvfile.write("957\n")