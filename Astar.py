import util
from util import PriorityQueue


class PriorityQueueItem:
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions

    def __lt__(self, other):
        return True
    score_ = 100
def a_star_search(problem, heuristic=None):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    fringe = util.PriorityQueue()
    fringe.push(PriorityQueueItem(problem, []), 0)
    visited = set()
    while not fringe.isEmpty():
        item = fringe.pop()
        if item.state.score > PriorityQueueItem.score_:
            print("score: ", item.state.score)
            PriorityQueueItem.score_ +=100
        state, actions = item.state, item.actions
        if state.is_goal_state():
            return actions, item.state.scorex
        if state not in visited:
            visited.add(state)
            for successor, action in state.get_successors():
                new_actions = actions + [action]
                new_cost = successor.score + heuristic(successor.grid)
                fringe.push_and_keep_n_largest(PriorityQueueItem(successor, new_actions), new_cost, 1000)
    return [], 0

import heapq

class PriorityQueue1:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def isEmpty(self):
        return len(self.heap) == 0

    def size(self):
        return len(self.heap)

class PriorityQueueItem1:
    def __init__(self, state, actions, cost):
        self.state = state
        self.actions = actions
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def a_star_search1(problem, heuristic=None, beam_width=500, max_cost=float('inf')):
    """
    A* search with beam search for limiting fringe size and optional max cost cutoff.
    - beam_width: limits how many nodes are expanded at each iteration
    - max_cost: ignores paths that exceed this threshold
    """
    fringe = PriorityQueue1()
    fringe.push(PriorityQueueItem1(problem, [], 0), 0)
    visited = set()

    while not fringe.isEmpty():
        # Limit fringe to top 'beam_width' nodes
        next_fringe = PriorityQueue1()

        for _ in range(min(beam_width, fringe.size())):
            item = fringe.pop()

            if item.state.is_goal_state():
                return item.actions, item.state.score

            if item.state not in visited:
                visited.add(item.state)

                for successor, action in item.state.get_successors():
                    new_actions = item.actions + [action]
                    new_cost = item.cost + successor.score + heuristic(successor.grid)

                    if new_cost <= max_cost:
                        next_fringe.push(PriorityQueueItem1(successor, new_actions, new_cost), new_cost)

        fringe = next_fringe

    return [], 0

import heapq
import util

class PriorityQueue2:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def push_and_keep_n_largest(self, item, priority, n):
        """Keep only the n largest elements in the queue"""
        self.push(item, priority)
        if len(self.heap) > n:
            heapq.heappop(self.heap)

def a_star_search2(problem, heuristic=None, beam_size=100, cost_cutoff=None):
    """
    Optimized A* search with beam search and optional cost cutoff.
    """
    fringe = PriorityQueue2()
    start_item = PriorityQueueItem(problem, [])
    fringe.push(start_item, 0)

    # Dictionary to track the cost of reaching each state
    cost_so_far = {problem: 0}

    visited = set()

    while not fringe.isEmpty():
        item = fringe.pop()

        state, actions = item.state, item.actions

        # Goal check
        if state.is_goal_state():
            return actions, state.score

        # Only expand if not visited or has a lower cost
        if state not in visited or cost_so_far[state] > state.score:
            visited.add(state)
            cost_so_far[state] = state.score

            # Explore successors
            for successor, action in state.get_successors():
                new_actions = actions + [action]
                new_cost = successor.score# + heuristic(successor.grid)

                # Apply optional cost cutoff to prune paths
                if cost_cutoff is None or new_cost <= cost_cutoff:
                    print("new_cost: ", new_cost)
                    successor_item = PriorityQueueItem(successor, new_actions)
                    fringe.push_and_keep_n_largest(successor_item, new_cost, beam_size)

    return [], 0  # Return empty if no solution found
