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
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# Heuristic function for the A* algorithm
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(parent, current, draw):
    while current in parent:
        current = parent[current]
        current.make_path()
        draw()


def A_star(draw, grid_map, start, end):
    entry_id = 0
    pq = PriorityQueue()
    pq.put((0, entry_id, start))
    parent = {}
    g_score = {grid: float("inf") for row in grid_map for grid in row}
    g_score[start] = 0
    f_score = {grid: float("inf") for row in grid_map for grid in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    pq_hash = {start}

    while not pq.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = pq.get()[2]
        pq_hash.remove(current)

        if current == end:
            reconstruct_path(parent, end, draw)
            end.make_end()
            return True

        ####################################################################
        # A* main loop
        ####################################################################
        for neighbor in current.neighbors:
            tentative_g = g_score[current] + 1  # Distance from start to neighbor through current

            if tentative_g < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor.get_pos(), end.get_pos())

                if neighbor not in pq_hash:
                    entry_id += 1
                    pq.put((f_score[neighbor], entry_id, neighbor))
                    pq_hash.add(neighbor)
                    neighbor.make_open()

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

                    A_star(lambda: draw(win, grid_map, ROWS, width), grid_map, start, end)
                    draw(win, grid_map, ROWS, width)
                    pygame.time.delay(500)  # Wait 0.5 second to make sure the screen updates

                    pygame.image.save(win, "pathfinding_result.png")
                    print("Saved the result to pathfinding_result.png!")

    pygame.quit()


if __name__ == '__main__':
    main(WIN, WIDTH)
