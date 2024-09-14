import util


def a_star_search(problem, heuristic=None):
    return a_star_helper(util.PriorityQueue(), problem, heuristic)

def a_star_helper(data_type, game, heuristic):
    data_type.push((game, []), 0)
    visited = set()

    while not data_type.isEmpty():
        state, path= data_type.pop()

        if state.is_goal_state():
            return path

        if state not in visited:
            visited.add(state)
            for successor, actionNode in state.get_successors():
                if successor not in visited:
                    new_path = path + [actionNode]
                    data_type.push((successor, new_path), -(successor.get_score() + heuristic(successor.grid)))
    return []