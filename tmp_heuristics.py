import random
import csv
from scipy.cluster.hierarchy import weighted


class Heuristics:

    def score(self, board):
        # Calculate the number of complete rows and cols
        rows = 0
        cols = 0
        for row in range(board.height):
            if all(board.grid[row]):
                rows += 1
        for col in range(board.width):
            if all([board.grid[row][col] for row in range(board.height)]):
                cols += 1
        return board.width * (rows + cols)

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
        empty_cells = 0
        for row in range(board.height):
            for col in range(board.width):
                if board.grid[row][col] == 0:
                    empty_cells += 1
        return empty_cells

    def calculate_smoothness(self, board):
        smoothness = 0
        for i in range(board.height):
            for j in range(board.height):
                if i + 1 < board.height:  # Compare vertically
                    smoothness -= abs(board.grid[i][j] - board.grid[i + 1][j])
                if j + 1 < len(board.grid[i]):  # Compare horizontally
                    smoothness -= abs(board.grid[i][j] - board.grid[i][j + 1])
        return smoothness

    def calculate_monotonicity(self, board):
        monotonicity = 0
        for i in range(board.height):
            row = board.grid[i]
            if row == sorted(row) or row == sorted(row, reverse=True):
                monotonicity += 1  # Row is monotonic
            col = [board.grid[j][i] for j in range(board.height)]
            if col == sorted(col) or col == sorted(col, reverse=True):
                monotonicity += 1  # Column is monotonic
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

    def sum_close_coordinates_values(self, board):
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

    def count_valid_moves(self, board):
        valid_moves = 0
        for i in range(len(board.grid)):
            for j in range(len(board.grid[i])):
                if i + 1 < len(board.grid) and board.grid[i][j] == board.grid[i + 1][j]:
                    valid_moves += 1
                if j + 1 < len(board.grid[i]) and board.grid[i][j] == board.grid[i][j + 1]:
                    valid_moves += 1
        return valid_moves

    def heuristic(self, board):
        # add weights to the different heuristics
        # score_weight = 1
        holes_weight = -10
        empty_cells_weight = 10
        smoothness_weight = 7
        monotonicity_weight = 0
        merges_weight = 0
        sum_close_coordinates_values_weight = 0
        count_valid_moves_weight = 0
        return (holes_weight * self.holes(board) +
                empty_cells_weight * self.empty_cells(board) +
                smoothness_weight * self.calculate_smoothness(board) +
                monotonicity_weight * self.calculate_monotonicity(board) +
                merges_weight * self.count_merge_opportunities(board) +
                sum_close_coordinates_values_weight * self.sum_close_coordinates_values(board) +
                count_valid_moves_weight * self.count_valid_moves(board)
                )
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