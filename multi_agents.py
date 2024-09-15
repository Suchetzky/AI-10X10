import random
import tracemalloc

import numpy as np
import abc
import util
from Game import Game, run_multiple_times
import abc

from heuristics import Heuristics
from util import Node as Action
import time
import pandas as pd

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
    # return current_game_state.score
    h = Heuristics()
    # return 0
    return h.heuristic(current_game_state.grid) + current_game_state.get_score()

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

    def __init__(self, evaluation_function='score_evaluation_function', depth=2):
        # self.evaluation_function = util.lookup(evaluation_function, globals())
        self.evaluation_function = evaluation_function
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
            return action, score_evaluation_function(game_state)
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
        return max_action, score

    def min_value(self, game_state, depth, param):
        score = float('inf')
        for game_state_, _ in game_state.get_successors():
            _, v = self.minimax(game_state_, depth - 1, 0)
            if v < score:
                score = v
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
            return Action.STOP,score_evaluation_function(game_state)
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
        return max_action, score


    def min_value(self, game_state, depth, max_player, alpha, beta):
        score = float('inf')
        for game_state_, _ in game_state.get_successors():
            _, v = self.alphabeta(game_state_, depth, alpha, beta, not max_player)
            if v < score:
                score = v
            beta = min(beta, score)
            if alpha >= beta:
                break
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
            return Action.STOP, score_evaluation_function(game_state)
        if max_player_flag:
            score = float('-inf')
            max_action = None
            for action in game_state.get_successors():
                _, v = self.expectimax(action[0], depth, not max_player_flag)
                if v > score:
                    score = v
                    max_action = action
            return max_action, score
        else:
            score = 0
            actions = game_state.get_successors()
            for action in actions:
                _, v = self.expectimax(action[0], depth - 1, not max_player_flag)
                score += v
            if len(actions) == 0:
                return None, score_evaluation_function(game_state)
            return None, score / len(actions)

def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return Heuristics.heuristic(current_game_state.grid)
    # heuristics = Heuristics(current_game_state)




class Game_runner(object):
    def __init__(self, agent=ReflexAgent(), opponent_agent=ReflexAgent(), display=None, sleep_between_actions=True, draw=True):
        super(Game_runner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        # self.display = display
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False
        self.draw = draw

    def run(self, initial_state):
        self._state = initial_state
        while not self._state.is_goal_state():
            action, score = self.agent.get_action(self._state)
            # print(self._state.get_score())
            if action is None or action[0].is_goal_state():
                return self._state.get_score()
            self._state.place_part_in_board_if_valid_by_shape(action) # apply action
            # if self.draw:
            #     self._state.draw()
            # opponent_action, _ = self.opponent_agent.get_action(self._state)
            # self._state.place_part_in_board_if_valid_by_shape(opponent_action) # apply opponent action todo check if this is correct
            # self.display.update_state(self._state, action, opponent_action)
            # if self.draw:
            #     self._state.draw()
        return self._state.get_score()


def run_multiple_times(game_instance, x):
    # List to store the results
    results = []

    for i in range(1, x + 1):
        # Recreate the game instance each time (to reset the game state)
        new_game_instance = game_instance.deepcopy()

        # Track memory and time for the current run
        time_taken, memory_used, score = track_memory_and_time_for_agent(new_game_instance)

        # Append the results (run number, time, memory)
        results.append([time_taken, memory_used, score])

    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=["Time Taken (seconds)", "Memory Used (MB)", "Score"])

    # Calculate the averages
    avg_time = df["Time Taken (seconds)"].mean()
    avg_memory = df["Memory Used (MB)"].mean()
    avg_score = df["Score"].mean()

    # Return the averages
    return avg_time, avg_memory, avg_score


def track_memory_and_time_for_agent(game_instance):
    # Start tracing memory allocations
    tracemalloc.start()

    # Start the timer to track the time for depth_first_search
    start_time = time.time()

    # Run depth_first_search
    agent = AlphaBetaAgent(depth=1)
    game_runner = Game_runner(agent, agent, draw=True)
    score = game_runner.run(game_instance)
    # print(score)
    # Stop the timer
    end_time = time.time()

    # Stop memory tracing and get the statistics
    current, peak = tracemalloc.get_traced_memory()

    # Stop tracing memory allocations
    tracemalloc.stop()

    # Return the time taken and peak memory usage
    return end_time - start_time, peak / 1024 / 1024, score  # time in seconds, memory in MB


# Abbreviation
better = better_evaluation_function


if __name__ == '__main__':
    print(Heuristics.random_weights())
    initial_game = Game(False, 10, 50, False)
    agent = AlphaBetaAgent(depth=1)
    game_runner = Game_runner(agent, agent, draw=True)
    avg_time, avg_memory, avg_score = run_multiple_times(initial_game, 1)
    print(f"Average Time Taken: {avg_time:.4f} seconds")
    print(f"Average Memory Used: {avg_memory:.4f} MB")
    print(f"Average Score: {avg_score:.4f}")

# [0, 1, -1, 0, 1, 0, 1]
# [0, 0, 1, -1, 0, -1, 1]
# Average Time Taken: 26.0149 seconds
# Average Memory Used: 0.5335 MB
# Average Score: 5450.0000
# [-1, 1, 0, -1, 1, 1, 1]
# Average Time Taken: 49.2861 seconds
# Average Memory Used: 0.5414 MB
# Average Score: 10969.0000
# 
