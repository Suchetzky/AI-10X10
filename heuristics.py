import random
import csv
import numpy as np


class Heuristics:
    _instance = None
    # holes_weight = 0
    # empty_cells_weight = 0
    # smoothness_weight = -1
    # monotonicity_weight = 0
    # merges_weight = 0
    # sum_close_coordinates_values_weight = 0
    # count_valid_moves_weight = 0

    holes_weight = random.randint(-10, 0)
    empty_cells_weight = random.randint(-10, 0)
    smoothness_weight = random.randint(0, 10)
    monotonicity_weight = random.randint(0, 10)
    merges_weight = random.randint(0, 10)
    # sum_close_coordinates_values_weight = random.randint(-10, 10)
    count_valid_moves_weight = random.randint(0, 10)
    blocks = 1
    weights = [holes_weight,
               empty_cells_weight,
               smoothness_weight,
               monotonicity_weight,
               merges_weight,
               # sum_close_coordinates_values_weight,
               count_valid_moves_weight
               ]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Heuristics, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    @classmethod
    def random_weights(cls):
        cls.holes_weight = random.randint(-1, 1)
        cls.empty_cells_weight = random.randint(-1, 1)
        cls.smoothness_weight = random.randint(-1, 1)
        cls.monotonicity_weight = random.randint(-1, 1)
        cls.merges_weight = random.randint(-1, 1)
        cls.count_valid_moves_weight = random.randint(-1, 1)
        cls.blocks = random.randint(-1, 1)

        # cls.holes_weight = 10.47743174
        # cls.empty_cells_weight =  30.83350653
        # cls.smoothness_weight = -12.93670911
        # cls.monotonicity_weight = 1.35595286
        # cls.merges_weight = 21.96272877
        # cls.count_valid_moves_weight = 19.15243476
        
        # cls.holes_weight = 0.45567818
        # cls.empty_cells_weight =  17.65677899
        # cls.smoothness_weight = 9.99443012
        # cls.monotonicity_weight = -10.71075892
        # cls.merges_weight = 2.9763484
        # cls.count_valid_moves_weight = -1.80218795
        # -1,-1,-1,-1,0,1
        # cls.holes_weight = -1
        # cls.empty_cells_weight = -1
        # cls.smoothness_weight = -1
        # cls.monotonicity_weight = -1
        # cls.merges_weight = 0
        # cls.count_valid_moves_weight = 1
        # -6.8106172   29.62203142   9.12958048 -15.96431503   6.7293583  1.6178059
        # cls.holes_weight = -6.8106172
        # cls.empty_cells_weight = 29.62203142
        # cls.smoothness_weight = 9.12958048
        # cls.monotonicity_weight = -15.96431503
        # cls.merges_weight = 6.7293583
        # cls.count_valid_moves_weight = 1.6178059
        cls.weights = [cls.holes_weight, cls.empty_cells_weight, cls.smoothness_weight, cls.monotonicity_weight, cls.merges_weight, cls.count_valid_moves_weight, cls.blocks]
        return cls.weights
        
    @staticmethod
    def heuristic2_2x2_blocks(board):
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
                        (row + 1 >= board.width or board.grid[row + 1][
                            col] == 1) and
                        (row - 1 <= 0 or board.grid[row - 1][col] == 1) and
                        (col - 1 <= 0 or board.grid[row][col - 1] == 1) and
                        (col + 1 >= board.height or board.grid[row][col + 1] == 1)):
                    holes += 1
        return holes

    # @staticmethod
    # def bumpiness_cols(board):
    #     # Calculate the bumpiness of the board
    #     # bumpiness = 0
    #     # for col in range(board.width - 1):
    #     #     bumpiness += abs(sum([board.grid[row][col] for row in
    #     #                           range(board.height)]) - sum(
    #     #         [board.grid[row][col + 1] for row in range(board.height)]))
    #     # return bumpiness
    #     col_sums = np.sum(board.grid, axis=0)
    #     return np.sum(np.abs(np.diff(col_sums)))
    # 
    # @staticmethod
    # def bumpiness_rows(board):
    #     # Calculate rows bumpiness
    #     # bumpiness = 0
    #     # for row in range(board.height):
    #     #     bumpiness += abs(sum(board.grid[row]) - sum(board.grid[row]))
    #     # return bumpiness
    #     row_sums = np.sum(board.grid, axis=1)
    #     return np.sum(np.abs(np.diff(row_sums)))

    @staticmethod
    def empty_cells(board):
        grid = np.array(board.grid)
        return np.sum(grid == 0)

    @staticmethod
    def calculate_smoothness(board):

        smoothness = 0
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=0)))  # Vertical smoothness
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=1)))  # Horizontal smoothness
        return smoothness

    @staticmethod
    def calculate_monotonicity(board):
        grid = np.array(board.grid)
        monotonicity = 0
        for row in grid:
            if np.all(np.diff(row) >= 0) or np.all(np.diff(row) <= 0):
                monotonicity += 1
        for col in grid.T:  # Transpose for columns
            if np.all(np.diff(col) >= 0) or np.all(np.diff(col) <= 0):
                monotonicity += 1
        return monotonicity


    @staticmethod
    def count_merge_opportunities(board):
        grid = np.array(board.grid)
        merges = 0
        # Vectorized checking for adjacent merges (rows)
        merges += np.sum(grid[:, :-1] == grid[:, 1:])
        # Vectorized checking for adjacent merges (columns)
        merges += np.sum(grid[:-1, :] == grid[1:, :])
        return merges


    @staticmethod
    def count_valid_moves(board):
        grid = np.array(board.grid)
        valid_moves = 0
        valid_moves += np.sum(grid[:, :-1] == grid[:, 1:])  # Horizontal
        valid_moves += np.sum(grid[:-1, :] == grid[1:, :])  # Vertical
        return valid_moves
    
    @staticmethod
    def heuristic1_adjacent_pairs(board):
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
    @staticmethod
    def large_shape_fit_heuristic(board):
        rows = len(board.grid)
        cols = len(board.grid[0])
        large_block_count = 0
    
        # Check for 3x3 block placement
        for i in range(rows - 2):
            for j in range(cols - 2):
                if Heuristics.can_place_block(board, i, j, 3, 3):
                    large_block_count += 1
    
        # Check for 5x1 block placement (horizontal)
        for i in range(rows):
            for j in range(cols - 4):
                if Heuristics.can_place_block(board, i, j, 1, 5):
                    large_block_count += 1
    
        # Check for 1x5 block placement (vertical)
        for i in range(rows - 4):
            for j in range(cols):
                if Heuristics.can_place_block(board, i, j, 5, 1):
                    large_block_count += 1
    
        return large_block_count
    
    def can_place_block(board, start_row, start_col, height, width):
        # Check if a block of size (height, width) can fit starting at (start_row, start_col)
        for i in range(start_row, start_row + height):
            for j in range(start_col, start_col + width):
                if board.grid[i][j] != 0:  # Assuming 0 represents an empty cell
                    return False
        return True


    class SubBoardHeuristic:
        # Example of how the preprocessed block probabilities might look:
        block_probabilities = {
            'completely_empty': 0.0,  # No chance of blocking when the sub-board is empty
            'completely_full': 1.0,   # Blocks definitely can't fit in a full sub-board
            'partially_filled': {
                15: 0.1,  # Probability for 15 empty cells
                14: 0.15, # Probability for 14 empty cells, and so on...
                # ... additional probabilities for other configurations ...
            }   
        }

        def __init__(self):
            self.block_probabilities = {
                'completely_empty': 0.0,  # No chance of blocking when the sub-board is empty
                'completely_full': 1.0,   # Blocks definitely can't fit in a full sub-board
                'partially_filled': {
                    15: 0.1,  # Probability for 15 empty cells
                    14: 0.15, # Probability for 14 empty cells, and so on...
                    # ... additional probabilities for other configurations ...
                }
            }  # Precomputed probabilities of blocks not fitting into a 4x4 sub-board

        def sub_board_analysis(self, board):
            rows = len(board.grid)
            cols = len(board.grid[0])
            total_probability = 0
    
            # Iterate over the board, analyzing each 4x4 sub-board
            for i in range(rows - 3):  # Rows where a 4x4 sub-board can fit
                for j in range(cols - 3):  # Columns where a 4x4 sub-board can fit
                    sub_board = self.get_sub_board(board, i, j)
                    total_probability += self.calculate_sub_board_probability(sub_board)
    
            return total_probability
    
        def get_sub_board(self, board, row, col):
            # Extract a 4x4 sub-board from the main board, starting at (row, col)
            sub_board = []
            for i in range(row, row + 4):
                sub_board.append(board.grid[i][col:col + 4])
            return sub_board
        
        def calculate_sub_board_probability(self, sub_board):
            # Check how likely it is that blocks won't fit into the current sub-board
            empty_cells = sum([row.count(0) for row in sub_board])  # Count empty cells in the sub-board
    
            # Based on the number of empty cells, look up the probability of blocks not fitting
            # You might define more detailed probabilities for specific block types in the preprocessing phase.
            if empty_cells == 16:
                return self.block_probabilities['completely_empty']
            elif empty_cells == 0:
                return self.block_probabilities['completely_full']
            else:
                # Compute probability for partially filled sub-board
                return self.block_probabilities['partially_filled'][empty_cells]


    @staticmethod
    def heuristic(board):
        # add weights to the different heuristics    
        return Heuristics.heuristic1_adjacent_pairs(board) * 0.124 + Heuristics.heuristic2_2x2_blocks(board)*0.186 + Heuristics.large_shape_fit_heuristic(board)*0.156 +Heuristics.SubBoardHeuristic().sub_board_analysis(board)*0.534
                
        # return (Heuristics.holes_weight * Heuristics.holes(board) +
        #                 Heuristics.empty_cells_weight * Heuristics.empty_cells(board) +
        #                 Heuristics.smoothness_weight * Heuristics.calculate_smoothness(board) +
        #                 Heuristics.monotonicity_weight * Heuristics.calculate_monotonicity(board) +
        #                 Heuristics.merges_weight * Heuristics.count_merge_opportunities(board) +
        #                 Heuristics.blocks * Heuristics.heuristic2_2x2_blocks(board) +
        #                 Heuristics.count_valid_moves_weight * Heuristics.count_valid_moves(board))
        # 
