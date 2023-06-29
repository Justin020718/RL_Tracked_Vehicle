import create_maze
import A_star_algorithm
import DDDQN
import PPO
import tkinter as tk
import test_consistence

if __name__ == '__main__':
    length = 32 # odd numbers only
    width = 32  # odd numbers only
    start = [1, 1]
    end = [length - 2, width - 2]
    grid_size = 20
    size = (length, width)
    maze_grid = create_maze.create_maze(size) # simple grid
    #maze_grid = create_maze.enlarge_path(maze_grid)
    #maze_grid = create_maze.create_continuous_maze(size) # more realistic grid
    window = tk.Tk()
    window.title('随机生成迷宫')
    window.geometry=(1366, 728)
    window.resizable=(1, 1)
    canvas = tk.Canvas(window, bg='#CDC9A5', width = length * grid_size, height = width * grid_size)
    canvas.pack()
    test_grid = create_maze.create_test(size)
    agent = 'PPO' # PPO or 3DQN or A*
    if agent == 'PPO':
        flag, route = PPO.solve(maze_grid, start, end)
    elif agent == '3DQN':
        flag, route = DDDQN.solve(maze_grid, start, end)
    elif agent == 'test_consistence':
        flag, route = test_consistence.solve(test_grid, start, end)
    else:
        flag, route = A_star_algorithm.solve(maze_grid, start, end)
    print(route)
    if flag == False:
        print('The maze is not connective, or the agent is stupid.')
    else:
        for i in range(length):
            for j in range(width):
                if route == 'Test':
                    if test_grid[i][j] == 1:
                        canvas.create_rectangle(grid_size * i, grid_size * j, grid_size * (i + 1), grid_size * (j + 1),
                                                fill='gray', outline='')
                else:
                    if maze_grid[i][j] == 1:
                        canvas.create_rectangle(grid_size*i, grid_size*j, grid_size*(i+1), grid_size*(j+1), fill = 'gray', outline='')
                    if (i,j) in route:
                        canvas.create_rectangle(grid_size*i, grid_size*j, grid_size*(i+1), grid_size*(j+1), fill = 'white', outline='')
    (a1, b1) = start
    (a2, b2) = end
    canvas.create_rectangle(grid_size * a1, grid_size * b1, grid_size * (a1 + 1), grid_size * (b1 + 1), fill='green', outline='')
    canvas.create_rectangle(grid_size * a2, grid_size * b2, grid_size * (a2 + 1), grid_size * (b2 + 1), fill='purple', outline='')
    window.mainloop()

