import random
import numpy as np


class Heuristics:
    _instance = None
    holes_weight = random.randint(-10, 0)
    empty_cells_weight = random.randint(-10, 0)
    smoothness_weight = random.randint(0, 10)
    monotonicity_weight = random.randint(0, 10)
    merges_weight = random.randint(0, 10)
    count_valid_moves_weight = random.randint(0, 10)
    blocks = 1
    weights = [holes_weight,
               empty_cells_weight,
               smoothness_weight,
               monotonicity_weight,
               merges_weight,
               count_valid_moves_weight]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Heuristics, cls).__new__(cls, *args, **kwargs)
        return cls._instance


    @classmethod
    def random_weights(cls):
        """
        This function is used for analysis score of the board
        :return:
        """
        cls.holes_weight = random.randint(-1, 1)
        cls.empty_cells_weight = random.randint(-1, 1)
        cls.smoothness_weight = random.randint(-1, 1)
        cls.monotonicity_weight = random.randint(-1, 1)
        cls.merges_weight = random.randint(-1, 1)
        cls.count_valid_moves_weight = random.randint(-1, 1)
        cls.blocks = random.randint(-1, 1)
        cls.weights = [cls.holes_weight, cls.empty_cells_weight,
                       cls.smoothness_weight, cls.monotonicity_weight,
                       cls.merges_weight, cls.count_valid_moves_weight,
                       cls.blocks]
        return cls.weights


    @staticmethod
    def blocks_heuristic(board):
        """
        Heuristic function that evaluates the quality of partials boards blocks,
        rewarding smoothness blocks-Adjacent filled/empty cells, penalizing holes.
        :param board:
        :return:
        """
        rows = len(board.grid)
        cols = len(board.grid[0])
        score = 0

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get the partials block
                block = [board.grid[i][j], board.grid[i][j + 1],
                         board.grid[i + 1][j],
                         board.grid[i + 1][j + 1]]
                occupied_count = block.count(
                    1)  # Assuming 1 represents filled, 0 represents empty

                if occupied_count == 4 or occupied_count == 0:
                    # Best score: all 4 are either filled or empty
                    score += 3
                elif (block[0] == block[1] and block[2] == block[3]) or (
                        block[0] == block[2] and block[1] == block[3]):
                    # Second-best score: One row or one column is fully filled or fully empty
                    score += 2
                elif occupied_count == 3 or occupied_count == 1:
                    # Third-best score: Three cells are filled and one is empty, or vice versa
                    score += 1
                elif occupied_count == 2:
                    # Worst score: every Adjacent cells are different
                    if (block[0] != block[1]) and (block[2] != block[3]) and (
                            block[0] != block[2]):
                        score -= 1

        return score


    @classmethod
    def write_weights_to_csv(cls, heuristic_value):
        """
        write the weights and heuristic value to a csv file, for data analysis
        :param heuristic_value:
        :return:
        """
        with open('data.csv', 'a') as csvfile:
            for i in range(len(Heuristics.weights)):
                csvfile.write(f"{Heuristics.weights[i]},")
            csvfile.write(f"{heuristic_value}\n")

    @staticmethod
    def holes(board):
        """
        Calculate the number of holes in the board.
        
        A hole is defined as an empty cell (represented by 0) that is surrounded by occupied cells (represented by 1) on all four sides (up, down, left, and right).
        
        Parameters:
        board (Board): The board object containing the grid to be evaluated.
        
        Returns:
        int: The total number of holes in the board.
        """
        # Calculate the number of holes in the board
        holes = 25
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
                    holes -= 1
        return holes


    @staticmethod
    def empty_cells(board):
        """
        Calculate the number of empty cells on the board.
        :param board:
        :return:
        """
        grid = np.array(board.grid)
        return np.sum(grid == 0)


    @staticmethod
    def calculate_smoothness(board):
        """
        Calculate the smoothness of the board
        :param board:
        :return:
        """
        smoothness = 0
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=0)))  # Vertical smoothness
        smoothness -= np.sum(
            np.abs(np.diff(board.grid, axis=1)))  # Horizontal smoothness
        return smoothness


    @staticmethod
    def calculate_monotonicity(board):
        """
        Calculate the monotonicity of the board
        :param board:
        :return:
        """
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
        """
        Count the number of merge opportunities on the board
        :param board:
        :return:
        """
        grid = np.array(board.grid)
        merges = 0
        # Vectorized checking for adjacent merges (rows)
        merges += np.sum(grid[:, :-1] == grid[:, 1:])
        # Vectorized checking for adjacent merges (columns)
        merges += np.sum(grid[:-1, :] == grid[1:, :])
        return merges


    @staticmethod
    def count_valid_moves(board):
        """
        Count the number of valid moves on the board
        :param board:
        :return:
        """
        grid = np.array(board.grid)
        valid_moves = 0
        valid_moves += np.sum(grid[:, :-1] == grid[:, 1:])  # Horizontal
        valid_moves += np.sum(grid[:-1, :] == grid[1:, :])  # Vertical
        return valid_moves


    @staticmethod
    def adjacent_pairs(board):
        """
        Check for adjacent pairs in the board
        :param board:
        :return:
        """
        rows = len(board.grid)
        cols = len(board.grid[0])
        score = 0

        # Check horizontal adjacent pairs
        for i in range(rows):
            for j in range(cols - 1):
                if board.grid[i][j] == board.grid[i][j + 1]:
                    score += 1

        # Check vertical adjacent pairs
        for i in range(rows - 1):
            for j in range(cols):
                if board.grid[i][j] == board.grid[i + 1][j]:
                    score += 1

        return score


    @staticmethod
    def large_shape_fit_heuristic(board):
        """
        Check if large blocks can fit in the board
        :param board:
        :return:
        """
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


    @staticmethod
    def can_place_block(board, start_row, start_col, height, width):
        """
        Check if a block of size (height, width) can fit starting at (start_row, start_col)
        :param board:
        :param start_row:
        :param start_col:
        :param height:
        :param width:
        :return:
        """
        # Check if a block of size (height, width) can fit starting at (start_row, start_col)
        for i in range(start_row, start_row + height):
            for j in range(start_col, start_col + width):
                if board.grid[i][j] != 0:  # Assuming 0 represents an empty cell
                    return False
        return True

    @staticmethod
    def snack(board):
        """
        Calculate the score based on consecutive 2x2 blocks that are either fully occupied or fully empty.
        
        This function iterates over the board grid and evaluates each 2x2 block of cells.
        It assigns a score based on the number of consecutive fully occupied or fully empty 2x2 blocks.
        
        Scoring rules:
        - Increment `score_full` by 1 for each consecutive 2x2 block that is fully occupied.
        - Increment `score_empty` by 1 for each consecutive 2x2 block that is fully empty.
        
        Parameters:
        board (Board): The board object containing the grid to be evaluated.
        
        Returns:
        int: The total score based on the evaluation of consecutive fully occupied or fully empty 2x2 blocks.
        """ 
        score_full = 1
        score_empty = 1
        prev_occupied_count = [board.grid[0][0], board.grid[0][1],
                               board.grid[1][0], board.grid[1][1]].count(1)
        for i in range(len(board.grid)-1):
            for j in range(1,len(board.grid[0])-1):
                block = [board.grid[i][j], board.grid[i][j + 1],
                         board.grid[i + 1][j],
                         board.grid[i + 1][j + 1]]
                occupied_count = block.count(1)
                if prev_occupied_count == 4 and occupied_count == 4:
                    score_full = score_full + 1
                elif prev_occupied_count == 0 and occupied_count == 0:
                    score_empty = score_empty + 1
                prev_occupied_count = occupied_count
        return score_full + score_empty

    @staticmethod
    def heuristic(board):
        return 0.5 * Heuristics.blocks_heuristic(board) + 0.25 * Heuristics.snack(board) + 0.25 * Heuristics.holes(board)
