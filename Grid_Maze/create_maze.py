import copy
import random
import numpy as np

def append_nearby(Var, a, b, length, width):
    #Var.append([a - 1, b - 1, a, b])
    #Var.append([a - 1, b + 1, a, b])
    #Var.append([a + 1, b - 1, a, b])
    #Var.append([a + 1, b + 1, a, b])
    Var.append([a, b + 1])
    Var.append([a, b - 1])
    Var.append([a - 1, b])
    Var.append([a + 1, b])
    for i in range(len(Var) - 1, - 1, - 1):
        (l, w) = Var[i]
        if l < 0 or w < 0 or l >= length - 1 or w >= width - 1:
            Var.remove([l, w])
    return Var

def break_permitted(wall, grid, viewed_blank):
    (x, y) = wall
    x_max = np.size(grid, 0)
    y_max = np.size(grid, 1)
    if x < 0 or y < 0 or x >= x_max-1 or y >= y_max-1:
        return False, -1, -1
    if grid[x-1][y] == 0 and [x-1, y] not in viewed_blank:
        viewed_blank.append([x-1, y])
        return True, x-1, y
    if grid[x+1][y] == 0 and [x+1, y] not in viewed_blank:
        viewed_blank.append([x+1, y])
        return True, x+1, y
    if grid[x][y-1] == 0 and [x, y-1] not in viewed_blank:
        viewed_blank.append([x, y-1])
        return True, x, y-1
    if grid[x][y+1] == 0 and [x, y+1] not in viewed_blank:
        viewed_blank.append([x, y+1])
        return True, x, y+1
    return False, -1, -1

def explore_next_state(grid, node):
    next_state = []
    (x, y) = node
    x_max = np.size(grid, 0)
    y_max = np.size(grid, 1)
    next_state.append(0)
    next_state.append(0)
    next_state.append(0)
    next_state.append(0)
    if  x+1 < 0 or y < 0 or x+1 >= x_max - 1 or y >= y_max - 1 or grid[x+1][y] == 1: # right
        next_state[3] = 1
    if  x-1 < 0 or y < 0 or x-1 >= x_max - 1 or y >= y_max - 1 or grid[x-1][y] == 1: # left
        next_state[2] = 1
    if  x < 0 or y+1 < 0 or x >= x_max - 1 or y+1 >= y_max - 1 or grid[x][y+1] == 1: # up
        next_state[1] = 1
    if  x < 0 or y-1 < 0 or x >= x_max - 1 or y-1 >= y_max - 1 or grid[x][y-1] == 1: # down
        next_state[0] = 1
    return next_state

def create_test(size):
    grid = np.ones(tuple(size))
    (length, width) = size
    grid[1][1] = 0
    grid[1][2] = 0
    return grid

def enlarge_path(grid):
    length = len(grid)
    width = len(grid[0])
    mapped_grid = np.zeros([length, width])
    for i in range(length):
        for j in range(width):
            mapped_grid[i][j] = 1 #unmapped element is 1
    for i in range(length):
        for j in range(width):
            if grid[i][j] == 0:
                if 2*i-1 < length and 2*j-1 < width:
                    mapped_grid[2*i-1][2*j-1] = 0
                if 2*i < length and 2*j-1<width:
                    mapped_grid[2*i][2*j-1] = 0
                if 2*i-1 < length and 2*j<width:
                    mapped_grid[2*i-1][2*j] = 0
                if 2*i < length and 2*j<width:
                    mapped_grid[2*i][2*j] = 0
    for i in range(0, length):
        for j in range(0, width):
            if i == 0 or j == 0 or i == length - 1 or j == length - 1:
                mapped_grid[i][j] = 1
    mapped_grid[length-2][width-2] = 0
    return mapped_grid


def create_maze(size):
    (_length, _width) = size
    length = int(_length//2+1)
    width = int(_width//2+1)

    grid = np.zeros(tuple([length, width]))
    grid_map = np.zeros(size)
    for i in range(0, length):
        for j in range(0, width):
            if i % 2 == 0 or j % 2 == 0 or i == length - 1 or j == width - 1:
                grid[i][j] = 1
    for i in range(0, _length):
        for j in range(0, _width):
            if i % 4 == 0 or j % 4 == 0 or i == _length - 1 or j == _width - 1:
                grid_map[i][j] = 1

    a = random.randint(0, length - 1)
    b = random.randint(0, width - 1)
    while grid[a][b] == 1:
        a = random.randint(0, length - 1)
        b = random.randint(0, width - 1)
    viewed_blank = []
    viewed_blank.append([a, b])
    viewed_wall = []
    viewed_wall = append_nearby(viewed_wall, a, b, length, width)
    while viewed_wall:
        wall = random.choice(viewed_wall)
        (a1, b1) = wall
        flag, x, y = break_permitted(wall, grid, viewed_blank)
        if flag:
            grid[a1][b1] = 0
            if 2 * a1 < _length - 1 and 2 * b1 < _width - 1:
                grid_map[2 * a1][2 * b1] = 0
            viewed_wall = append_nearby(viewed_wall, x, y, length, width)
        viewed_wall.remove(wall)
    _grid = copy.deepcopy(grid_map)
    for i in range(0, _length):
        for j in range(0, _width):
            if grid_map[i][j] == 1:
                if sum(explore_next_state(grid_map,[i,j])) <= 1:
                    _grid[i][j] =0
    __grid = copy.deepcopy(_grid)
    for i in range(0, length):
        for j in range(0, width):
            if _grid[i][j] == 1:
                if sum(explore_next_state(_grid,[i,j])) <= 1:
                    __grid[i][j] =0
    ___grid = copy.deepcopy(__grid)
    for i in range(0, length):
        for j in range(0, width):
            if __grid[i][j] == 1:
                if sum(explore_next_state(__grid,[i,j])) <= 1:
                    ___grid[i][j] =0
    return __grid

def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)
    return lerp(x1,x2,v)

def lerp(a,b,x):
    return a + x * (b-a)

def fade(t):
    " 6t^5 - 15t^4 + 10t^3 "
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    " grad converts h to the right gradient vector and return the dot product with (x,y) "
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def create_continuous_maze(size):
    (length, width) = size
    linx = np.linspace(0, length/10, length, endpoint=False)
    liny = np.linspace(0, width/10, width, endpoint=False)
    x, y = np.meshgrid(liny, linx)
    grid = perlin(x, y, seed = 0) # some seeds may creat disconnected grid， which is time consuming! seed 0, 1 and 3 are good.
    for i in range(0, length):
        for j in range(0, width):
            if i == 0 or j == 0 or i == length - 1 or j == width - 1:
                grid[i][j] = 1
            if grid[i][j] <= 0.1:
                grid[i][j] = 0
            else:
                grid[i][j] = 1
    grid[1][1] = 0
    grid[length-2][width-2] = 0
    return grid

