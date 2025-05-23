import pygame
import math
import numpy as np
from queue import PriorityQueue


WIDTH = 700
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Planning Visualizer")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

np.random.seed(47)

# Represents each grid in the grid map.
class Grid:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
           self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
           self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
           self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
           self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False



def reconstruct_path(prev, current, draw):
    while current in prev:
        current = prev[current]
        current.make_path()
        draw()



def Dijkstra(draw, grid_map, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    prev = {}
    d_v = {grid: float("inf") for row in grid_map for grid in row}
    d_v[start] = 0

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(prev, end, draw)
            end.make_end()
            return True
        ####################################################################
        # TODO: Implement your Dijkstra algorithm here
        # 
        # We assume the distances between two adjacent grids are 1.
        #
        # You will need to use data structure "PriorityQueue" to store the open grid
        # The functions of PriorityQueue you need to understand and use are:
        # put()
        # get()
        # 
        # You will need to use open_set_hash to track grid open and closed status
        # 
        # You will need to understand and use the following functions 
        # in proper position of you path planning algorithm 
        # to visualize the result:
        #   a.  make_open
        #   b.  make_closed
        #   c.  make_end
        #   d.  draw
        #   e.  reconstruct_path
        ####################################################################
        for neighbor in current.neighbors:
            temp_dist = d_v[current] + 1  # Assuming all edges have weight = 1

            if temp_dist < d_v[neighbor]:
                prev[neighbor] = current
                d_v[neighbor] = temp_dist
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((d_v[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

 

        ####################################################################
        # End TODO
        ####################################################################

        draw()

        if current != start:
            current.make_closed()

    return False





def grid_map_gen(rows, width, obstacle_density=0.2):
    print('rows: ', rows)
    k = int(obstacle_density * rows * rows)
    mask = np.random.choice(list(range(rows * rows - 1)), k, replace=False)

    grid_map = []
    gap = width // rows
    for i in range(rows):
        grid_map.append([])
        for j in range(rows):
            grid = Grid(i, j, gap, rows)
            if i * rows + j in mask:
                grid.make_barrier()
            grid_map[i].append(grid)

    return grid_map

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
        pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def main(win, width):
    ROWS = 50
    grid_map = grid_map_gen(ROWS, width)

    start = None
    end = None

    run = True
    while run:
        draw(win, grid_map, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                grid = grid_map[row][col]
                if not start and grid != end:
                    start = grid
                    start.make_start()

                elif not end and grid != start:
                    end = grid
                    end.make_end()


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid_map:
                        for grid in row:
                            grid.update_neighbors(grid_map)

                Dijkstra(lambda: draw(win, grid_map, ROWS, width), grid_map, start, end)



    pygame.quit()


if __name__ == '__main__':
    main(WIN, WIDTH)
