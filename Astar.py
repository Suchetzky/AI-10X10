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
            return actions, item.state.score
        if state not in visited:
            visited.add(state)
            for successor, action in state.get_successors():
                new_actions = actions + [action]
                new_cost = successor.score + heuristic(successor.grid)
                fringe.push_and_keep_n_largest(PriorityQueueItem(successor, new_actions), new_cost, 1000)
    return [], 0

