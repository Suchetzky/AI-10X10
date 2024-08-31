import tkinter as tk
from tkinter import font as tkFont
from Grid import Grid
from Shape import Shape
from util import Stack, Queue
from util import Node

# Set up display dimensions
screen_width, screen_height = 500, 550

class Game:
    def __init__(self, NoUI=False, board_len=10, size=50, test = False):
        self.board_len = board_len
        self.size = size
        self.grid = Grid(board_len, board_len, size)
        self.next_shapes = []
        if not test:
            self.add_next_shapes()
        else:
            self.next_shapes = [Shape(specific_shape_num=0), Shape(specific_shape_num=16), Shape(specific_shape_num=9)]
        self.placed_pieces = 0
        self.piece_num = 0
        self.current_x = 0
        self.current_y = 0
        self.score = 0
        self.headless = NoUI # game is runninf with or without graphical interface
        self.running = True
        if not self.headless:
            self._setup_ui()

    ########## UI SETUP AND DRAWING ##########

    def _setup_ui(self):
        self.root = tk.Tk()
        self.root.title('10x10 Game')
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, bg='black')
        self.canvas.pack()
        self.font = tkFont.Font(family="Helvetica", size=20)
        self.root.bind('<Left>', self.move_left)
        self.root.bind('<Right>', self.move_right)
        self.root.bind('<Down>', self.move_down)
        self.root.bind('<Up>', self.move_up)
        self.root.bind('<r>', self.rotate_piece)
        self.root.bind('<space>', self.place_piece)

    def draw(self):
        if self.headless:
            return
        self.canvas.delete('all')
        self.grid.draw(self.canvas)
        if self.next_shapes:
            self.next_shapes[self.piece_num].draw(self.canvas, self.current_x, self.current_y, self.grid.size)
        self.canvas.create_text(self.board_len, screen_height - 40, anchor='nw', text=f'Score: {self.score}', fill='white', font=self.font)
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
        if self.grid.can_place(self.next_shapes[self.piece_num].shape, self.current_x + dx, self.current_y + dy):
            self.current_x += dx
            self.current_y += dy
            self.draw()

    def rotate_piece(self, event): # todo change shape?
        if self.next_shapes:
            self.piece_num = (self.piece_num + 1) % len(self.next_shapes)
            self.draw()

    def place_piece(self, event):
        self.place_part_in_board_if_valid(self.current_x, self.current_y)

    ########## GAME LOGIC ##########

    def add_next_shapes(self):
        while len(self.next_shapes) < 3:
            new_shape = Shape()
            while new_shape in self.next_shapes:
                new_shape = Shape()
            self.next_shapes.append(new_shape)
    # get coordinates and check if the placement of self.next_shapes[self.piece_num] is valid
    # if so, place the shape in the board and clear lines if needed and update the score
    def place_part_in_board_if_valid(self, x, y): # todo check
        if self.grid.can_place(self.next_shapes[self.piece_num].shape, x, y, check_placement=True):
            self.score += self.next_shapes[self.piece_num].get_part_size()
            self.grid.place_shape(self.next_shapes[self.piece_num].shape, x, y)
            self.score += self.grid.clear_lines()
            self.next_shapes.pop(self.piece_num)
            self.placed_pieces += 1
            if self.placed_pieces == 3:
                self.add_next_shapes()
                self.placed_pieces = 0
                self.piece_num = 0
            else:
                self.piece_num = min(self.piece_num, len(self.next_shapes) - 1)
            self.current_x, self.current_y = 0, 0
            self.draw()

    def has_valid_placement(self):
        for shape in self.next_shapes:
            if self.get_board_available_places(shape.shape):
                return True
        return False

    def get_board_available_places(self, shape): # for bfs/dfs
        available_places = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(shape, x, y, check_placement=True):
                    available_places.append((x, y))
        return available_places

    def is_game_over(self):
        return not self.has_valid_placement()

    def display_game_over(self):
        if self.headless:
            return
        self.canvas.delete('all')
        game_over_text = "Game Over"
        score_text = f'Score: {self.score}'
        self.canvas.create_text(screen_width // 2, screen_height // 2 - 50, text=game_over_text, fill='white', font=self.font)
        self.canvas.create_text(screen_width // 2, screen_height // 2, text=score_text, fill='white', font=self.font)
        self.root.update()
        self.root.after(3000, self.root.quit)

    def run(self):
        while self.running:
            self.draw()
            if self.is_game_over():
                self.display_game_over()
                self.running = False
        self.root.mainloop()

    def run_from_code(self, nodes):
        for node in nodes:
            self.piece_num = node.action[2]
            self.next_shapes = node.action[3]
            self.place_part_in_board_if_valid(node.action[0], node.action[1])
            if not self.headless: # todo: what is this?
                self.root.after(1000, self.root.update_idletasks)
        if self.is_game_over():
            self.display_game_over()
        return self.score

    ########## HELPER FUNCTIONS ##########

    def get_board(self):
        return self.grid.grid

    def get_score(self):
        return self.score

    def get_next_shapes(self):
        return [shape.shape for shape in self.next_shapes]

    # get all valid placements for a shape
    def get_valid_placements(self, shape):
        valid_placements = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(shape, x, y):
                    valid_placements.append((x, y))
        return valid_placements
    def get_valid_placements(self):
        valid_placements = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(self.next_shapes[self.piece_num].shape, x, y):
                    valid_placements.append((x, y))
        return valid_placements

    def get_successors(self):
        successors = []
        if len(self.next_shapes) == 0: # just if we finish the shapes batch we will randomly add 3 new shapes
            self.add_next_shapes() # todo: why its always 3?
        for piece_num in range(len(self.next_shapes)):
            for x, y in self.get_board_available_places(self.next_shapes[piece_num].shape):
                new_game = self.deepcopy()
                new_game.piece_num = piece_num
                new_game.place_part_in_board_if_valid(x, y)
                successors.append((new_game, Node((x, y, piece_num, self.next_shapes)))) # todo : why tuple?
        # print(len(successors)) # todo remove
        return successors

    def is_goal_state(self):
        return self.score >= 100

    def deepcopy(self):
        new_game = Game(NoUI=True, board_len=self.board_len, size=self.size)
        new_game.grid = Grid(self.board_len, self.board_len, self.size)
        new_game.grid.grid = [row[:] for row in self.grid.grid]
        new_game.next_shapes = self.next_shapes.copy()
        new_game.piece_num = self.piece_num
        new_game.current_x = self.current_x
        new_game.current_y = self.current_y
        new_game.score = self.score
        return new_game

    def test(self):
        self.grid.place_shape(Shape.shapes[9], 0, 0)
        for row in self.grid.grid:
            print(row)
        print("\n")
        for row in self.get_board():
            print(row)
        self.place_part_in_board_if_valid(0, 0)

        print(self.get_board())



######################## DFS BFS ########################
def depth_first_search(problem):
    stack = Stack()
    return bfs_dfs_helper(stack, problem)

def breadth_first_search(problem):
    queue = Queue()
    return bfs_dfs_helper(queue, problem)

def bfs_dfs_helper(data_type, game): # todo check
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

# def print_path(grid):
#     for node in grid:
#         for row in node:
#             print(row)
#         print("\n")
#         # print(node, "\n")





if __name__ == '__main__':
    initial_game = Game(False, 4, 50, False) # false- no UI, 5- board size, 50- cell size
    # initial_game.test()
    # initial_game.run() # run the game
    solution_path, grid = depth_first_search(initial_game)
    # print_path(grid)
    print("Solution Path:", solution_path)
    initial_game.run_from_code(solution_path) # run the game with the solution path
