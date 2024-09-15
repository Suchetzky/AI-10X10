import time

import util


def a_star_search(problem, heuristic=None, start_time=None):
    data_type = util.PriorityQueue()    
    data_type.push((problem, []), 0)
    
    visited = set()

    while not data_type.isEmpty():
        # if time.time() - start_time > 30:
        #     return []
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

# 1,-2,-4,1,1,1,5.476709842681885
# 2,0,-5,-7,4,3,3.797492027282715
# 6,-9,-3,5,10,2,139.1242082118988
# -10,0,-4,10,3,1,4.841771125793457
# 9,4,-10,-4,3,0,818.7896101474762
# -7,-3,6,9,4,7,33.91989302635193
# -2,9,-8,8,5,5,821.8306310176849
# 4,-9,-9,-8,-10,10,14.851457118988037
# -5,-9,-6,4,10,7,50.963995695114136
# 3,-3,9,-6,7,2,40.92701315879822
# 7,3,-6,10,-4,7,9.784566164016724
# -6,-4,-6,-4,9,0,9.13992714881897
# -1,-8,7,8,4,7,54.231656074523926
# -10,-9,5,5,9,0,84.58693981170654
# -4,-2,9,4,9,5,139.0248429775238
# -2,-5,7,10,3,9,35.60375499725342
# -10,-4,3,2,8,9,29.336066007614136
# -10,-1,8,1,0,10,153.38968920707703
# 0,-4,5,4,0,8,21.025875091552734
# -2,0,6,3,1,9,72.95520997047424
# -4,-4,5,4,6,10,47.0757269859314
# 0,-2,2,10,3,4,15.128910064697266