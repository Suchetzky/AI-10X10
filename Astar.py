import util


def a_star_search(problem, heuristic=None, start_time=None):
    """
    Perform A* search to find the optimal path to the goal state.

    :param problem: The initial problem or state to solve, typically an object that contains the initial state,
                    goal conditions, and methods to retrieve successors (neighboring states).
    :param heuristic: A function that estimates the cost from the current state to the goal (default is None).
    :param start_time: Optional parameter, can be used to track time (not utilized in the current function).

    :return: A list of actions representing the path to the goal, or an empty list if no path is found.
    """
    # Priority queue to store (state, path) with priority determined by the sum of the cost and heuristic
    data_type = util.PriorityQueue()
    # Push the initial problem with an empty path and a priority of 0 (initial state cost is zero)
    data_type.push((problem, []), 0)

    # Set to track visited states to avoid re-exploration
    visited = set()

    # Main loop: continue until the priority queue is empty (no more states to explore)
    while not data_type.isEmpty():
        # Pop the state with the lowest priority (highest priority, lowest cost)
        state, path = data_type.pop()

        # Check if the current state is the goal state, if so, return the path
        if state.is_goal_state():
            return path

        # If the state hasn't been visited yet, mark it as visited
        if state not in visited:
            visited.add(state)

            # Get successors (neighboring states) from the current state
            for successor, actionNode in state.get_successors():
                if successor not in visited:
                    # Construct the new path by appending the action to the current path
                    new_path = path + [actionNode]
                    # Calculate the priority using the successor's score and the heuristic function
                    priority = -(successor.get_score() + heuristic(successor.grid))
                    # Push the successor and its path to the priority queue with the calculated priority
                    data_type.push((successor, new_path), priority)

    # Return an empty list if no goal state is found after exploring all possible states
    return []
