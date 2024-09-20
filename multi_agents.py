import abc
import time
import tracemalloc

import numpy as np

from Game import Game
from heuristics import Heuristics
from util import Node as Action
import time
import pandas as pd

"""
# Agent class to implement the agents
"""
class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass

"""
# ReflexAgent class 
"""
class ReflexAgent(Agent):
    def get_action(self, game_state):
        # Collect legal moves and successor states
        legal_moves = game_state.get_successors()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action_stat, action) for
                  action_stat, action in legal_moves]
        if len(scores) == 0:
            return Action.STOP, self.evaluation_function(game_state, game_state,
                                                         None)
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if
                        scores[index] == best_score]
        chosen_index = np.random.choice(
            best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index], best_score

    """
    # Helper function to evaluate the score of the game state
    # @param current_game_state: The current game state
    # @return: The score of the heuristic function and the current game state
    """
    def evaluation_function(self, action_stat):
        return action_stat.get_score()

"""
# Helper function to evaluate the score of the game state
# @param current_game_state: The current game state
# @return: The score of the heuristic function and the current game state
"""
def score_evaluation_function(current_game_state):
    h = Heuristics()
    return h.heuristic(current_game_state.grid) + current_game_state.get_score()

"""
# MultiAgentSearchAgent class to implement the agents
"""
class MultiAgentSearchAgent(Agent):

    def __init__(self, evaluation_function='score_evaluation_function',
                 depth=2):
        # self.evaluation_function = util.lookup(evaluation_function, globals())
        self.evaluation_function = evaluation_function
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return

"""
# Minmax agent
"""
class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        return self.minimax(game_state, self.depth, 0)

    def minimax(self, game_state, depth, param, action=None):
        if depth == 0:
            return action, score_evaluation_function(game_state)
        if param == 0:
            return self.max_value(game_state, depth, param)
        else:
            return self.min_value(game_state, depth, param)

    def max_value(self, game_state, depth):
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            _, v = self.minimax(action[0], depth - 1, 1)
            if v > score:
                score = v
                max_action = action
        return max_action, score

    def min_value(self, game_state, depth):
        score = float('inf')
        for game_state_, _ in game_state.get_successors():
            _, v = self.minimax(game_state_, depth - 1, 0)
            if v < score:
                score = v
        return None, score

"""
# AlphaBeta agent
"""
class AlphaBetaAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        return self.alphabeta(game_state, self.depth, -np.inf, np.inf)

    def alphabeta(self, game_state, depth, alpha, beta, max_player=True,
                  action=None):
        if depth == 0:
            return Action.STOP, score_evaluation_function(game_state)
        if max_player:
            return self.max_value(game_state, depth, max_player, float('-inf'),
                                  float('inf'))
        else:
            return self.min_value(game_state, depth, max_player, float('-inf'),
                                  float('inf'))

    def max_value(self, game_state, depth, max_player, alpha, beta):
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            _, v = self.alphabeta(action[0], depth - 1, alpha, beta,
                                  not max_player)
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
            _, v = self.alphabeta(game_state_, depth, alpha, beta,
                                  not max_player)
            if v < score:
                score = v
            beta = min(beta, score)
            if alpha >= beta:
                break
        return None, score

"""
# Expectimax agent
"""
class ExpectimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
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
                _, v = self.expectimax(action[0], depth - 1,
                                       not max_player_flag)
                score += v
            if len(actions) == 0:
                return None, score_evaluation_function(game_state)
            return None, score / len(actions)

"""
# Greedy agent
"""
class GreedyAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        return self.greedy(game_state, self.depth)

    def greedy(self, game_state, depth):
        if depth == 0:
            return Action.STOP, score_evaluation_function(game_state)
        score = float('-inf')
        max_action = None
        for action in game_state.get_successors():
            if action[0].get_score() > 100 and score_evaluation_function(
                    action[0]) < score - 100:
                continue
            _, v = self.greedy(action[0], depth - 1)
            if v > score:
                score = v
                max_action = action
        return max_action, score

"""
# Game runner class to run the game
"""
class Game_runner(object):
    def __init__(self, agent=ReflexAgent(), opponent_agent=ReflexAgent(),
                 sleep_between_actions=True, draw=True):
        super(Game_runner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        self.opponent_agent = opponent_agent
        self._state = None
        self.draw = draw

    """
    # Run the game with agents
    # @param initial_state: The initial state of the game
    # @return: The score of the game
    """
    def run(self, initial_state):
        self._state = initial_state
        while not self._state.is_goal_state():
            action, score = self.agent.get_action(self._state)
            if action is None or action[0].is_goal_state():
                return self._state.get_score()
            self._state.place_part_in_board_if_valid_by_shape(
                action)  # apply action
            if self.draw:
                self._state.draw()
            if self.sleep_between_actions:
                time.sleep(1)
            opponent_action, _ = self.opponent_agent.get_action(self._state)
            self._state.place_part_in_board_if_valid_by_shape(
                opponent_action)  # apply opponent action
            if self.draw:
                self._state.draw()
            if self.sleep_between_actions:
                time.sleep(1)
        return self._state.get_score()

"""
# Helper function to run the game multiple times and collect the results
# @param game_instance: The game instance to run the agent on
# @param x: The number of times to run the game
# @return: The average time taken and peak memory usage
"""
def run_multiple_times(game_instance, x):
    # List to store the results
    results = []

    for i in range(1, x + 1):
        # Recreate the game instance each time (to reset the game state)
        new_game_instance = game_instance.deepcopy()

        # Track memory and time for the current run
        time_taken, memory_used, score = track_memory_and_time_for_agent(
            new_game_instance)

        # Append the results (run number, time, memory)
        results.append([time_taken, memory_used, score])

    # Convert results to a DataFrame
    df = pd.DataFrame(results,
                      columns=["Time Taken (seconds)", "Memory Used (MB)",
                               "Score"])

    # Calculate the averages
    avg_time = df["Time Taken (seconds)"].mean()
    avg_memory = df["Memory Used (MB)"].mean()
    avg_score = df["Score"].mean()

    # Return the averages
    return avg_time, avg_memory, avg_score


"""
# Helper function to track memory and time for the agent
# @param game_instance: The game instance to run the agent on
# @return: The time taken and peak memory usage
"""
def track_memory_and_time_for_agent(game_instance):
    # Start tracing memory allocations
    tracemalloc.start()

    # Start the timer to track the time for depth_first_search
    start_time = time.time()

    # Run depth_first_search
    agent = GreedyAgent(depth=2)
    opponent_agent = GreedyAgent(depth=2)
    game_runner = Game_runner(agent, opponent_agent, draw=True)
    score = game_runner.run(game_instance)
    # Stop the timer
    end_time = time.time()

    # Stop memory tracing and get the statistics
    current, peak = tracemalloc.get_traced_memory()

    # Stop tracing memory allocations
    tracemalloc.stop()

    # Return the time taken and peak memory usage
    return end_time - start_time, peak / 1024 / 1024, score  # time in seconds, memory in MB


if __name__ == '__main__':
    # File path for the Excel file
    file_path = 'results.xlsx'

    # List to store results
    results = []

    # Simulate your game and collect results
    initial_game = Game(False, 10, 50, False)
    for i in range(1):
        avg_time, avg_memory, avg_score = run_multiple_times(initial_game, 1)
        # Collect the results in a dictionary (for easier DataFrame conversion)
        print({
            'Average Time Taken (s)': round(avg_time, 4),
            'Average Memory Used (MB)': round(avg_memory, 4),
            'Average Score': round(avg_score, 4)
        })
