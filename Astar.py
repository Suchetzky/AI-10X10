
import util


def a_star_search(problem, heuristic=None, start_time=None):
    """
    A* search algorithm
    :param problem:
    :param heuristic:
    :param start_time:
    :return:
    """
    data_type = util.PriorityQueue()    
    data_type.push((problem, []), 0)
    
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