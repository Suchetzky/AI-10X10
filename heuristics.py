import random
import csv
from scipy.cluster.hierarchy import weighted


class Heuristics:
    @staticmethod
    def __init__(self):
        # add weights to the different heuristics
        self.score_weight = random.randint(0, 10)
        self.holes_weight = random.randint(-10, 0)
        self.bumpiness_cols_weight = random.randint(-10, 0)
        self.bumpiness_rows_weight = random.randint(-10, 0)
        self.blocks_of_shapes_weight = random.randint(0, 15)
        # write the aggregation of the heuristics to the data.csv file
        with open('data.csv', 'a') as csvfile:
            # fieldnames = ['score', 'holes', 'bumpiness_cols', 'bumpiness_rows', 'results']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            csvfile.write(
                f"{self.score_weight},{self.holes_weight},{self.bumpiness_cols_weight},{self.bumpiness_rows_weight},")

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
        
    def holes(self, board):
        # Calculate the number of holes in the board
        holes = 0
        for col in range(board.width):
            for row in range(board.height):
                if board.grid[row][col] == 1:
                    continue
                if self.sum_close_coordinates_values(board) > 3:
                    holes += 1
        return holes
    
    def blocks_of_shapes(self, board):
        # Calculate the number of blocks of shapes in the board
        blocks = 0
        for col in range(board.width):
            for row in range(board.height):
                # If the cell is empty and the cells around are full
                if board.grid[row][col] == 0:
                    continue
                if self.sum_close_coordinates_values(board) >= 2:
                    blocks += 1
        return blocks

    def bumpiness_cols(self, board):
        # Calculate the bumpiness of the board
        bumpiness = 0
        for col in range(board.width - 1):
            bumpiness += abs(sum([board.grid[row][col] for row in range(board.height)]) - sum(
                [board.grid[row][col + 1] for row in range(board.height)]))
        return bumpiness

    def bumpiness_rows(self, board):
        # Calculate rows bumpiness
        bumpiness = 0
        for row in range(board.height - 1):
            bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row + 1]))
        return bumpiness

    def heuristic(self, board):
        return (self.score_weight * self.score(board) +
                self.holes_weight * self.holes(board) +
                self.bumpiness_cols_weight * self.bumpiness_cols(board) +
                self.bumpiness_rows_weight * self.bumpiness_rows(board) +
                self.blocks_of_shapes_weight * self.blocks_of_shapes(board))


if __name__ == '__main__':
    # Test the heuristics
    from Grid import Grid

    grid = Grid(4, 4, 50)
    heuristics = Heuristics.heuristic(grid)

    print(heuristics.heuristic())
    with open('data.csv', 'a') as csvfile:
        csvfile.write("9000\n")

    print(heuristics.heuristic(grid))

    with open('data.csv', 'a') as csvfile:
        csvfile.write("957\n")
