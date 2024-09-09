import util

class PriorityQueueItem:
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions

    def __lt__(self, other):
        return True

def a_star_search(problem, heuristic=None):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    fringe = util.PriorityQueue()
    fringe.push(PriorityQueueItem(problem.get_start_state(), []), 0)
    visited = set()
    while not fringe.isEmpty():
        item = fringe.pop()
        state, actions = item.state, item.actions
        if problem.is_goal_state(state):
            return actions
        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.get_successors(state):
                new_actions = actions + [action]
                new_cost = problem.get_cost_of_actions(new_actions) + heuristic(successor, problem)
                fringe.push(PriorityQueueItem(successor, new_actions), new_cost)
    return []