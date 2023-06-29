import  numpy as np

def heuristic(start, end):
    (x, y) = start
    (x1, y1) = end
    return abs(x1 - x) + abs(y1 - y)

def get_next_state(grid, node):
    next_state = []
    (x, y) = node
    x_max = np.size(grid, 0)
    y_max = np.size(grid, 1)
    next_state.append([x - 1, y])
    next_state.append([x + 1, y])
    next_state.append([x, y - 1])
    next_state.append([x, y + 1])
    for i in range(len(next_state)-1, -1, -1):
        (x1, y1) = next_state[i]
        if  grid[x1][y1] == 1 or x1 < 0 or y1 < 0 or x >= x_max - 1 or y >= y_max - 1:
            next_state.remove([x1, y1])
    return next_state

def solve(grid, start, end):
    if start == end:
        return True, None
    route = {}
    viewed_node = []
    init_f = heuristic(start, end)
    queue = {tuple(start):init_f} # dic
    cost = {tuple(start):0}
    route_node = []
    while True:
        if queue == {}:
            return False, None
        node = min(queue, key=queue.get)
        queue.pop(node)
        if node == tuple(end):
            node1 = route[tuple(end)]
            while node1 != tuple(start):
                route_node.append(node1)
                node1 = route[node1]
            return True, route_node # , route
        viewed_node.append(node)
        next_states = get_next_state(grid, node)

        for next_state in next_states:
            cost[tuple(next_state)] = cost[tuple(node)] + 1
            if tuple(next_state) not in queue or tuple(next_state) not in viewed_node:
                queue[tuple(next_state)] = heuristic(next_state, end) + cost[tuple(next_state)]
            elif tuple(next_state) in queue and heuristic(next_state, end) + cost[tuple(next_state)] < queue[tuple(next_state)]:
                queue[tuple(next_state)] = heuristic(next_state, end) + cost[tuple(next_state)]
                #route[tuple(next_state)] = node
            if tuple(next_state) not in route:
                route[tuple(next_state)] = node

