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
        # Calculate the number of empty cells
        # empty_cells = 0
        # for row in range(board.height):
        #     for col in range(board.width):
        #         if board.grid[row][col] == 0:
        #             empty_cells += 1
        # return empty_cells
        grid = np.array(board.grid)
        return np.sum(grid == 0)

    def calculate_smoothness(self, board):
        smoothness = 0
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

    def calculate_monotonicity(self, board):
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

    def heuristic(self, board, weights):
        return (weights['count_valid_moves_weight'] * self.count_valid_moves(board)
                + weights['holes_weight'] * self.holes(board)
                + weights['empty_cells_weight'] * self.empty_cells(board)
                + weights['smoothness_weight'] * self.calculate_smoothness(board)
                + weights[
                    'monotonicity_weight'] * self.calculate_monotonicity(
            board)
                + weights['merges_weight'] * self.count_merge_opportunities(
            board)
                + weights['bumpiness_weight'] * (self.bumpiness_cols(
            board) + self.bumpiness_rows(board))
                + weights['corner_weight'] * self.corner_heuristic(board)
                + weights['edge_weight'] * self.edge_heuristic(board)
        )

    # def heuristic(self, board):
    #     # add weights to the different heuristics
    #     # holes_weight = -10
    #     # empty_cells_weight = 15
    #     # smoothness_weight = 10
    #     # monotonicity_weight = 0
    #     # merges_weight = 0
    #     count_valid_moves_weight = 1
    #     holes_weight = -1
    #     empty_cells_weight = 0
    #     smoothness_weight = 0
    #     monotonicity_weight = 0
    #     merges_weight = 0
    #     corner_weight = 0
    #     edge_weight = 0
    #     bumpiness_weight = -0
    #     # count_valid_moves_weight = 4  # Reward finding valid moves more strongly
    #     # holes_weight = -6  # Keep holes penalty high to prevent trapping blocks
    #     # empty_cells_weight = 8  # Encourage empty cells for more flexibility
    #     # smoothness_weight = 5  # Reward smoother transitions between adjacent tiles
    #     # monotonicity_weight = 4  # Encourage keeping numbers increasing in a row/column
    #     # merges_weight = 10  # Prioritize merging tiles for higher values
    #     # corner_weight = 3  # Some reward for putting high-value tiles in corners
    #     # edge_weight = 2  # Mild reward for placing tiles along edges
    #     # bumpiness_weight = -3  # Penalize bumpiness, but not too heavily
    #
    #     # print(self.calculate_smoothness(board))
    #
    #     return (
    #             count_valid_moves_weight * self.count_valid_moves(board)
    #             + holes_weight * self.holes(board)
    #             + empty_cells_weight * self.empty_cells(board)
    #             + smoothness_weight * self.calculate_smoothness(board)
    #             + monotonicity_weight * self.calculate_monotonicity(board)
    #             + merges_weight * self.count_merge_opportunities(board)
    #             + bumpiness_weight * (self.bumpiness_cols(board)+ self.bumpiness_rows(board))
    #             + corner_weight * self.corner_heuristic(board)
    #             + edge_weight * self.edge_heuristic(board)
    #             + merges_weight * self.count_merge_opportunities(board)
    #             )

        # return (holes_weight * self.holes(board) +
        #         empty_cells_weight * self.empty_cells(board) +
        #         smoothness_weight * self.calculate_smoothness(board) +
        #         monotonicity_weight * self.calculate_monotonicity(board)
        #         + merges_weight * self.count_merge_opportunities(board)
        #         + sum_close_coordinates_values_weight * self.sum_close_coordinates_values(board)
        #         + count_valid_moves_weight * self.count_valid_moves(board)
        #         # + 4 * self.border_heuristic(board)
        #         )
        # bumpiness_cols_weight = random.randint(-10, 5)
        # bumpiness_rows_weight = random.randint(-10, 5)
        # write the aggregation of the heuristics to the data.csv file
        # with open('data.csv', 'a') as csvfile:
        #     # fieldnames = ['score', 'holes', 'bumpiness_cols', 'bumpiness_rows', 'results']
        #     # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     # writer.writeheader()
        #     csvfile.write(f"{score_weight},{holes_weight},{bumpiness_cols_weight},{bumpiness_rows_weight},")

        # return holes_weight * self.holes(board) + bumpiness_cols_weight * self.bumpiness_cols(board) + bumpiness_rows_weight * self.bumpiness_rows(board)




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