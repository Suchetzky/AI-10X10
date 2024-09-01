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
                        (row+1 >= board.width or board.grid[row + 1][col] == 1) and
                        (row-1 <= 0 or board.grid[row - 1][col] == 1) and
                        (col-1 <= 0 or board.grid[row][col - 1] == 1) and
                        (col+1 >= board.height or board.grid[row][col + 1] == 1)):
                    holes += 1
        return holes
    
    def bumpiness_cols(self, board):
        # Calculate the bumpiness of the board
        bumpiness = 0
        for col in range(board.width - 1):
            bumpiness += abs(sum([board.grid[row][col] for row in range(board.height)]) - sum([board.grid[row][col + 1] for row in range(board.height)]))
        return bumpiness
    
    def bumpiness_rows(self, board):
        # Calculate rows bumpiness
        bumpiness = 0
        for row in range(board.height):
            bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row]))
        return bumpiness
    
    def heuristic(self, board):
        # add weights to the different heuristics
        score_weight = random.randint(0, 10)
        holes_weight = random.randint(-10, 5)
        bumpiness_cols_weight = random.randint(-10, 5)
        bumpiness_rows_weight = random.randint(-10, 5)
        # write the aggregation of the heuristics to the data.csv file
        with open('data.csv', 'a') as csvfile:
            # fieldnames = ['score', 'holes', 'bumpiness_cols', 'bumpiness_rows', 'results']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            csvfile.write(f"{score_weight},{holes_weight},{bumpiness_cols_weight},{bumpiness_rows_weight},")

        return score_weight * self.score(board) + holes_weight * self.holes(board) + bumpiness_cols_weight * self.bumpiness_cols(board) + bumpiness_rows_weight * self.bumpiness_rows(board)
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