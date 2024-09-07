import numpy as np
import abc
import util
from Game import Game
import abc
from util import Node as Action
import time

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_successors()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action_stat, action) for action_stat, action in legal_moves]
        if len(scores) == 0:
            return Action.STOP, self.evaluation_function(game_state, game_state, None)
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index], best_score

    def evaluation_function(self, current_game_state, action_stat, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        return action_stat.get_score()


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        # self.evaluation_function = util.lookup(evaluation_function, globals())
        self.evaluation_function = score_evaluation_function
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        # util.raiseNotDefined()
        return self.minimax(game_state, self.depth, 0)

    def minimax(self, game_state, depth, param, action=None):
        if depth == 0:
            return action, self.evaluation_function(game_state)
        if param == 0:
            return self.max_value(game_state, depth, param)
        else:
            return self.min_value(game_state, depth, param)

    def max_value(self, game_state, depth, param):
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            _, v = self.minimax(action[0], depth - 1, 1)
            if v > score:
                score = v
                max_action = action
        if score == float('-inf'):
            return Action.STOP, self.evaluation_function(game_state)
        return max_action, score

    def min_value(self, game_state, depth, param):
        score = float('inf')
        for game_state_, _ in game_state.get_successors():
            _, v = self.minimax(game_state_, depth - 1, 0)
            if v < score:
                score = v
        if score == float('inf'):
            return None, self.evaluation_function(game_state)
        return None, score



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        return self.alphabeta(game_state, self.depth, -np.inf, np.inf)

    def alphabeta(self, game_state, depth, alpha, beta, max_player=True, action=None):
        if depth == 0:
            return Action.STOP, self.evaluation_function(game_state)
        if max_player:
            return self.max_value(game_state, depth, not max_player, float('-inf'), float('inf'))
        else:
            return self.min_value(game_state, depth, max_player, float('-inf'), float('inf'))

    def max_value(self, game_state, depth, max_player, alpha, beta):
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            _, v = self.alphabeta(action[0], depth - 1, alpha, beta, not max_player)
            if v > score:
                score = v
                max_action = action
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        if score == float('-inf'):
            return Action.STOP, self.evaluation_function(game_state)
        return max_action, score


    def min_value(self, game_state, depth, max_player, alpha, beta):
        score = float('inf')
        for game_state_, _ in game_state.get_successors():
            _, v = self.alphabeta(game_state_, depth - 1, alpha, beta, not max_player)
            if v < score:
                score = v
            beta = min(beta, score)
            if alpha >= beta:
                break
        if score == float('inf'):
            return None, self.evaluation_function(game_state)
        return None, score



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        return self.expectimax(game_state, self.depth, True)

    def expectimax(self, game_state, depth, max_player_flag):
        if depth == 0:
            return Action.STOP, self.evaluation_function(game_state)
        if max_player_flag:
            return self.max_value(game_state, depth, not max_player_flag)
        else:
            return self.exp_value(game_state, depth, max_player_flag)

    def max_value(self, game_state, depth, param):
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            _, v = self.expectimax(action[0], depth - 1, 1)
            if v > score:
                score = v
                max_action = action
        if score == float('-inf'):
            return Action.STOP, self.evaluation_function(game_state)
        return max_action, score

    def exp_value(self, game_state, depth, max_player_flag):
        pass


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



class Game_runner(object):
    def __init__(self, agent=ReflexAgent(), opponent_agent=ReflexAgent(), display=None, sleep_between_actions=True):
        super(Game_runner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        # self.display = display
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False

    def run(self, initial_state):
        self._should_quit = False
        self._state = initial_state
        # self.display.initialize(initial_state)
        return self._game_loop()

    def quit(self):
        self._should_quit = True
        self.agent.stop_running()
        self.opponent_agent.stop_running()

    def _game_loop(self):
        score = 0
        actions = []
        while not self._state.is_goal_state() and not self._should_quit:
            # if self.sleep_between_actions:
            #     time.sleep(2)
            # self.display.mainloop_iteration()
            action, score = self.agent.get_action(self._state)
            actions.append(action)
            # print(score)
            if action is None or action[0].is_goal_state():
                return score
            self._state.place_part_in_board_if_valid_by_shape(action) # apply action
            opponent_action, _ = self.opponent_agent.get_action(self._state)
            self._state.place_part_in_board_if_valid_by_shape(opponent_action) # apply opponent action todo check if this is correct
            # self.display.update_state(self._state, action, opponent_action)
            self._state.draw()
        return score



# Abbreviation
better = better_evaluation_function

if __name__ == '__main__':
    initial_game = Game(False, 10, 50, False)
    agent = MinmaxAgent()
    game_runner = Game_runner(agent, agent)
    score = game_runner.run(initial_game)
    print(score)
    # initial_game.run_from_code(solution_path)
    # agent.get_action(initial_game)
