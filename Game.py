import argparse
import tkinter as tk
from tkinter import font as tkFont
import Astar
from Grid import Grid
from Shape import Shape
from heuristics import Heuristics
from util import Stack, Queue
from util import Node
import time
import tracemalloc
import multi_agents
import pandas as pd

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
BACKGROUND_COLOR = 'black'
TEXT_COLOR = 'white'
FONT_FAMILY = 'Helvetica'
FONT_SIZE = 20
WINDOW_TITLE = '10x10 Game'
GAME_OVER_TEXT = 'Game Over'
GAME_OVER_DELAY_MS = 3000
SCORE_TEXT_OFFSET_Y = 40
GAME_OVER_TEXT_OFFSET_Y = 50
DEFAULT_BOARD_LEN = 10
DEFAULT_SIZE = 50
DEFAULT_GOAL = 10000


class Game:
    def __init__(self, NoUI=False, board_len=DEFAULT_BOARD_LEN, size=DEFAULT_SIZE, test=False,
                 sleep_between_actions=False, goal_state=DEFAULT_GOAL):
        self.board_len = board_len
        self.size = size
        self.grid = Grid(board_len, board_len, size)
        self.next_shapes = []
        if not test:
            self.add_next_shapes()
        else:
            self.next_shapes = [Shape(specific_shape_num=0),
                                Shape(specific_shape_num=16),
                                Shape(specific_shape_num=9)]
        self.sleep_between_actions = sleep_between_actions
        self.placed_pieces = 0
        self.piece_num = 0
        self.current_x = 0
        self.current_y = 0
        self.score = 0
        self.headless = NoUI  # game is running with or without graphical interface
        self.running = True
        self.goal_state = goal_state
        if not self.headless:
            self._setup_ui()

    ########## UI SETUP AND DRAWING ##########

    def _setup_ui(self):
        """Set up the game UI."""
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH,
                                height=SCREEN_HEIGHT, bg=BACKGROUND_COLOR)
        self.canvas.pack()
        self.font = tkFont.Font(family=FONT_FAMILY, size=FONT_SIZE)
        self.root.bind('<Left>', self.move_left)
        self.root.bind('<Right>', self.move_right)
        self.root.bind('<Down>', self.move_down)
        self.root.bind('<Up>', self.move_up)
        self.root.bind('<r>', self.rotate_piece)
        self.root.bind('<space>', self.place_piece)

    def draw(self):
        """Draw the game."""
        if self.headless:
            return
        self.canvas.delete('all')
        self.grid.draw(self.canvas)
        if self.next_shapes:
            self.next_shapes[self.piece_num].draw(self.canvas, self.current_x,
                                                  self.current_y,
                                                  self.grid.size)
            for i in range(len(self.next_shapes)):
                self.next_shapes[i].draw(self.canvas, 11, (i * 6),
                                         self.grid.size)
        self.canvas.create_text(self.board_len, SCREEN_HEIGHT - SCORE_TEXT_OFFSET_Y, anchor='nw',
                                text=f'Score: {self.score}', fill=TEXT_COLOR,
                                font=self.font)
        self.root.update()

    ########## GAME MOVEMENT ##########

    def move_left(self, event):
        self._move_piece(-1, 0)

    def move_right(self, event):
        self._move_piece(1, 0)

    def move_down(self, event):
        self._move_piece(0, 1)

    def move_up(self, event):
        self._move_piece(0, -1)

    def _move_piece(self, dx, dy):
        """Move the current piece.

        Args:
            dx (int): Change in x-coordinate.
            dy (int): Change in y-coordinate.
        """
        if self.grid.can_place(self.next_shapes[self.piece_num].shape,
                               self.current_x + dx, self.current_y + dy):
            self.current_x += dx
            self.current_y += dy
            self.draw()

    def rotate_piece(self, event):
        """Rotate the current piece."""
        if self.next_shapes:
            self.piece_num = (self.piece_num + 1) % len(self.next_shapes)
            self.draw()

    def place_piece(self, event):
        """Place the current piece on the board if the placement is valid."""
        self.place_part_in_board_if_valid(self.current_x, self.current_y)

    ########## GAME LOGIC ##########

    def add_next_shapes(self):
        """Add the next set of shapes to the game."""
        while len(self.next_shapes) < 3:
            new_shape = Shape()
            while new_shape in self.next_shapes:
                new_shape = Shape()
            self.next_shapes.append(new_shape)

    def place_part_in_board_if_valid_by_shape(self, action):
        """Attempt to place the shape specified by action on the board.

        Checks if the placement of the shape is valid; if so, places the shape,
        clears lines if needed, and updates the score.

        Args:
            action: The action containing the placement coordinates and piece info.
        """
        if action is None:
            return
        x, y, piece_num, next_shapes = action[1].action
        shape = next_shapes[piece_num]
        self._place_shape_if_valid(shape, x, y)

    def place_part_in_board_if_valid(self, x, y):
        """Attempt to place the current shape at the given coordinates.

        Checks if the placement is valid; if so, places the shape, clears lines if needed,
        and updates the score.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.
        """
        shape = self.next_shapes[self.piece_num]
        self._place_shape_if_valid(shape, x, y)

    def _place_shape_if_valid(self, shape, x, y):
        """Helper method to place a shape if the placement is valid.

        Args:
            shape (Shape): The shape to place.
            x (int): The x-coordinate.
            y (int): The y-coordinate.
        """
        if self.grid.can_place(shape.shape, x, y, check_placement=True):
            self.score += shape.get_part_size()
            self.grid.place_shape(shape.shape, x, y)
            self.score += self.grid.clear_lines()
            if shape in self.next_shapes:
                self.next_shapes.remove(shape)
            self.placed_pieces += 1
            if self.placed_pieces == 3:
                self.add_next_shapes()
                self.placed_pieces = 0
                self.piece_num = 0
            else:
                self.piece_num = min(self.piece_num, len(self.next_shapes) - 1)
            self.current_x, self.current_y = 0, 0

    def has_valid_placement(self):
        """Check if there is a valid placement for any of the next shapes.

        Returns:
            bool: True if there is a valid placement, False otherwise.
        """
        for shape in self.next_shapes:
            if self.get_board_available_places(shape.shape):
                return True
        return False

    def get_board_available_places(self, shape):
        """Get all available places to place a shape on the board.

        Args:
            shape: The shape to place.

        Returns:
            list of tuple: A list of (x, y) coordinates where the shape can be placed.
        """
        available_places = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(shape, x, y, check_placement=True):
                    available_places.append((x, y))
        return available_places

    def get_empty_cells(self):
        """Get the number of empty cells in the grid.

        Returns:
            int: The number of empty cells.
        """
        empty_cells = 0
        for row in self.grid.grid:
            for cell in row:
                if cell == 0:
                    empty_cells += 1
        return empty_cells

    def is_game_over(self):
        """Check if the game is over."""
        return not self.has_valid_placement()

    def display_game_over(self):
        """Display the game over screen."""
        if self.headless:
            return
        self.canvas.delete('all')
        game_over_text = GAME_OVER_TEXT
        score_text = f'Score: {self.score}'
        self.canvas.create_text(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - GAME_OVER_TEXT_OFFSET_Y,
                                text=game_over_text, fill=TEXT_COLOR,
                                font=self.font)
        self.canvas.create_text(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                                text=score_text, fill=TEXT_COLOR, font=self.font)
        self.root.update()
        self.root.after(GAME_OVER_DELAY_MS, self.root.quit)

    def run(self):
        """Run the game."""
        while self.running:
            self.draw()
            if self.is_game_over():
                self.display_game_over()
                self.running = False
        self.root.mainloop()

    def run_from_code(self, nodes):
        """Run the game from a list of nodes (for DFS, BFS, and A* algorithms).

        Args:
            nodes (list): A list of nodes to run.

        Returns:
            int: The final score.
        """
        for node in nodes:
            self.piece_num = node.action[2]
            self.next_shapes = node.action[3]
            self.draw()
            self.place_part_in_board_if_valid(node.action[0], node.action[1])
            if self.sleep_between_actions:  # break between steps
                time.sleep(1)
            if not self.headless:
                self.root.after(1000, self.root.update_idletasks)
        if self.is_game_over():
            self.display_game_over()
        return self.score

    ########## HELPER FUNCTIONS ##########

    def get_board(self):
        """Get the current board state.

        Returns:
            list: The current board state.
        """
        return self.grid.grid

    def get_score(self):
        """Get the current score.

        Returns:
            int: The current score.
        """
        return self.score

    def get_next_shapes(self):
        """Get the next shapes.

        Returns:
            list: A list of the next shapes.
        """
        return [shape.shape for shape in self.next_shapes]

    def get_valid_placements_for_shape(self, shape):
        """Get all valid placements for a shape.

        Args:
            shape: The shape to place.

        Returns:
            list of tuple: A list of (x, y) coordinates where the shape can be placed.
        """
        valid_placements = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(shape, x, y):
                    valid_placements.append((x, y))
        return valid_placements

    def get_valid_placements(self):
        """Get valid placements for the current shape.

        Returns:
            list of tuple: A list of (x, y) coordinates where the current shape can be placed.
        """
        valid_placements = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(self.next_shapes[self.piece_num].shape,
                                       x, y):
                    valid_placements.append((x, y))
        return valid_placements

    def get_successors(self):
        """Get the successors of the current state.

        Returns:
            list of tuple: Each tuple contains a new game instance and the action taken to reach that state.
        """
        successors = []
        if len(self.next_shapes) == 0:
            self.add_next_shapes()
        for piece_num in range(len(self.next_shapes)):
            for x, y in self.get_board_available_places(
                    self.next_shapes[piece_num].shape):
                new_game = self.deepcopy()
                new_game.piece_num = piece_num
                new_game.place_part_in_board_if_valid(x, y)
                successors.append((new_game, Node(
                    (x, y, piece_num, self.next_shapes))))
        return successors

    def is_goal_state(self):
        """Check if the current state is a goal state.

        Returns:
            bool: True if the current state is a goal state, False otherwise.
        """
        return self.score >= self.goal_state

    def deepcopy(self):
        """Create a deep copy of the game.

        Returns:
            Game: A new game instance with the same state as the current game.
        """
        new_game = Game(NoUI=True, board_len=self.board_len, size=self.size)
        new_game.grid = Grid(self.board_len, self.board_len, self.size)
        new_game.grid.grid = [row[:] for row in self.grid.grid]
        new_game.next_shapes = self.next_shapes.copy()
        new_game.piece_num = self.piece_num
        new_game.current_x = self.current_x
        new_game.current_y = self.current_y
        new_game.score = self.score
        new_game.sleep_between_actions = self.sleep_between_actions
        new_game.goal_state = self.goal_state
        return new_game

    ###### test ######
    def test(self):
        """Test function."""
        self.grid.place_shape(Shape.shapes[9], 0, 0)
        for row in self.grid.grid:
            print(row)
        print("\n")
        for row in self.get_board():
            print(row)
        self.place_part_in_board_if_valid(0, 0)
        print(self.get_board())

    #### for agents ####
    def __lt__(self, other):
        """Less-than comparison based on the score.

        Args:
            other (Game): Another game instance.

        Returns:
            bool: True if self's score is less than other's score.
        """
        return self.get_score() < other.get_score()

    #############################


######################## DFS BFS ########################
def depth_first_search(problem):
    """Perform depth-first search.

    Args:
        problem: The problem to solve.

    Returns:
        tuple: The solution path and the grid.
    """
    stack = Stack()
    return bfs_dfs_helper(stack, problem)


def breadth_first_search(problem):
    """Perform breadth-first search.

    Args:
        problem: The problem to solve.

    Returns:
        tuple: The solution path and the grid.
    """
    queue = Queue()
    return bfs_dfs_helper(queue, problem)


def bfs_dfs_helper(data_type, game):
    """Helper function for BFS and DFS.

    Args:
        data_type: The data structure to use (Stack or Queue).
        game: The game to solve.

    Returns:
        tuple: The solution path and the grid.
    """
    data_type.push((game, [], []))
    visited = set()

    while not data_type.isEmpty():
        state, path, grid = data_type.pop()
        if state.is_goal_state():
            return path, grid
        if state not in visited:
            visited.add(state)
            for successor, actionNode in state.get_successors():
                if successor not in visited:
                    new_path = path + [actionNode]
                    new_grid = [state.get_board()] + [successor.get_board()]
                    data_type.push((successor, new_path, new_grid))
    return [], []


########################################################

### For data collection
def track_memory_and_time_(game_instance):
    """Track memory and time for collecting data.

    Args:
        game_instance: The game instance to solve.

    Returns:
        tuple: The time taken and peak memory usage.
    """
    Heuristics.random_weights()
    tracemalloc.start()
    start_time = time.time()
    solution_path, grid = depth_first_search(game_instance)
    solution_path = Astar.a_star_search(game_instance, Heuristics.heuristic)
    initial_game.run_from_code(solution_path)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    Heuristics.write_weights_to_csv(end_time - start_time)
    return end_time - start_time, peak / 1024 / 1024  # time in seconds, memory in MB


def run_multiple_times(game_instance):
    """Run multiple times for data collection.

    Args:
        game_instance: The game instance to solve.

    Returns:
        tuple: The average time taken and peak memory usage.
    """
    results = []
    for i in range(1, args.num_of_games + 1):
        new_game_instance = game_instance.deepcopy()
        time_taken, memory_used = track_memory_and_time_(new_game_instance)
        results.append([time_taken, memory_used])
    df = pd.DataFrame(results,
                      columns=["Time Taken (seconds)", "Memory Used (MB)"])
    avg_time = df["Time Taken (seconds)"].mean()
    avg_memory = df["Memory Used (MB)"].mean()
    print(f"Average Time Taken: {avg_time:.4f} seconds")
    print(f"Average Memory Used: {avg_memory:.4f} MB")
    return avg_time, avg_memory


def get_parser():
    """Get arguments from the user.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    global args
    parser = argparse.ArgumentParser(description='10x10 Game')
    formats = ['play', 'DFS', 'A_star', 'agent', 'BFS']
    agents = ['NextMoveMaximizerAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']
    parser.add_argument("--display",
                        help="The game UI. True for GUI False otherwise",
                        type=str, default='True')
    parser.add_argument("--format",
                        help="choose format: 'play', 'DFS', 'A_star', 'agents', 'BFS'",
                        type=str, default='play', choices=formats)
    parser.add_argument('--agent', choices=agents,
                        help='The agent. default is AlphaBetaAgent',
                        default=agents[1], type=str)
    parser.add_argument('--depth',
                        help='The maximum depth for to search in the game tree.',
                        default=1, type=int)
    parser.add_argument('--sleep_between_actions',
                        help='Should sleep between actions.', default='False',
                        type=str)
    parser.add_argument('--score_goal',
                        help='The score goal to reach. for DFS and A_star',
                        default=10000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    initial_game = Game(False, 10, 50, False,
                        args.sleep_between_actions, goal_state=args.score_goal)
    initial_game.headless = True if args.display == 'False' else False
    initial_game.sleep_between_actions = True if args.sleep_between_actions == 'True' else False
    initial_game.goal_state = args.score_goal
    agent = multi_agents.AlphaBetaAgent(depth=args.depth)
    solution_path = []
    if args.agent == 'NextMoveMaximizerAgent':
        agent = multi_agents.NextMoveMaximizerAgent()
    elif args.agent == 'AlphaBetaAgent':
        agent = multi_agents.AlphaBetaAgent(depth=args.depth)
    elif args.agent == 'ExpectimaxAgent':
        agent = multi_agents.ExpectimaxAgent(depth=args.depth)
    score = 0
    tracemalloc.start()
    start_time = time.time()
    current, peak = None, None
    if args.format == 'play':
        initial_game.headless = False
        initial_game.run()
    elif args.format == 'DFS':
        solution_path, grid = depth_first_search(initial_game)
    elif args.format == 'A_star':
        solution_path = Astar.a_star_search(initial_game, Heuristics.heuristic)
    elif args.format == 'agent':
        game_runner = multi_agents.GameRunner(agent, agent, draw=args.display,
                                              sleep_between_actions=initial_game.sleep_between_actions)
        score = game_runner.run(initial_game)
    elif args.format == 'BFS':
        solution_path, grid = breadth_first_search(initial_game)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if args.display and args.format != 'play':
        initial_game.run_from_code(solution_path)
    time_taken = end_time - start_time
    memory_used = peak / 1024 / 1024
    print(f"Time Taken: {time_taken:.4f} seconds")
    print(f"Peak Memory Usage: {memory_used:.4f} MB")
    if args.format == 'agent':
        print(f"Score: {score}")
