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
        item, score = fringe.pop()
        if item.state.score > PriorityQueueItem.score_:
            print("score: ", item.state.score)
            PriorityQueueItem.score_ +=100
        state, actions = item.state, item.actions
        if problem.is_goal_state():
            return actions
        if state not in visited:
            visited.add(state)
            for successor, action in problem.get_successors():
                new_actions = actions + [action]
                new_cost = successor.score + heuristic(successor.grid)
                fringe.push(PriorityQueueItem(successor, new_actions), new_cost)
    return []

