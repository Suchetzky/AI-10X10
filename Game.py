import pygame
from Grid import Grid
from Shape import Shape

# Initialize the game
pygame.init()

# Set up display
screen_width, screen_height = 500, 550
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('10x10 Game')

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)

class Game:
    def __init__(self):
        self.grid = Grid(10, 10, 50)
        self.current_shape = Shape()
        self.next_shapes = [Shape() for _ in range(3)]
        self.current_x = 0
        self.current_y = 0
        self.score = 0
        self.font = pygame.font.SysFont(None, 35)
        self.running = True

    def draw(self):
        screen.fill(black)
        self.grid.draw(screen)
        self.current_shape.draw(screen, self.current_x, self.current_y, self.grid.size)
        score_text = self.font.render(f'Score: {self.score}', True, white)
        screen.blit(score_text, (10, screen_height - 40))
        pygame.display.flip()

    def has_valid_placement(self):
        return bool(self.get_board_available_places(self.current_shape.shape))

    def get_board_available_places(self, shape):
        available_places = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(shape, x, y, check_placement=True):
                    available_places.append((x, y))
        return available_places

    def is_game_over(self):
        return not self.has_valid_placement()

    def get_valid_placements(self):
        valid_placements = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.can_place(self.current_shape.shape, x, y):
                    valid_placements.append((x, y))
        return valid_placements


    def display_game_over(self):
        screen.fill(black)
        game_over_text = self.font.render('Game Over', True, white)
        score_text = self.font.render(f'Score: {self.score}', True, white)
        screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, screen_height // 2 - 50))
        screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, screen_height // 2))
        pygame.display.flip()
        pygame.time.delay(3000)

    def run(self):
        while self.running:
            self.draw()

            if self.is_game_over():
                self.display_game_over()
                self.running = False
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.grid.can_place(self.current_shape.shape, self.current_x - 1, self.current_y):
                        self.current_x -= 1
                    elif event.key == pygame.K_RIGHT and self.grid.can_place(self.current_shape.shape, self.current_x + 1, self.current_y):
                        self.current_x += 1
                    elif event.key == pygame.K_DOWN and self.grid.can_place(self.current_shape.shape, self.current_x, self.current_y + 1):
                        self.current_y += 1
                    elif event.key == pygame.K_UP and self.grid.can_place(self.current_shape.shape, self.current_x, self.current_y - 1):
                        self.current_y -= 1
                    elif event.key == pygame.K_r:
                        rotated_shape = list(zip(*self.current_shape.shape[::-1]))
                        if self.grid.can_place(rotated_shape, self.current_x, self.current_y):
                            self.current_shape.shape = rotated_shape
                    elif event.key == pygame.K_SPACE:
                        if self.grid.can_place(self.current_shape.shape, self.current_x, self.current_y, check_placement=True):
                            self.grid.place_shape(self.current_shape.shape, self.current_x, self.current_y)
                            self.score += self.grid.clear_lines()
                            self.current_shape = self.next_shapes.pop(0)
                            self.next_shapes.append(Shape())
                            self.current_x, self.current_y = 0, 0

            pygame.time.delay(100)


if __name__ == '__main__':
    game = Game()
    game.run()
    pygame.quit()