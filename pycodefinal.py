import matplotlib.pyplot as plt
import heapq
import time
import math
import random

GRID_SIZE = 10
NUM_OBSTACLES = 15
start = (0, 0)  # 좌상단 출발
goal = (GRID_SIZE - 1, GRID_SIZE - 1)  # 우하단 목표

def generate_obstacles():
    obstacles = set()
    while len(obstacles) < NUM_OBSTACLES:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        if (x, y) != start and (x, y) != goal:
            obstacles.add((x, y))
    return obstacles

def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def neighbors(node, obstacles):
    x, y = node
    candidates = [
        (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1)
    ]
    valid = []
    for nx, ny in candidates:
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in obstacles:
            # 대각선 이동 시 코너 장애물 체크
            if abs(nx - x) == 1 and abs(ny - y) == 1:
                if (nx, y) in obstacles or (x, ny) in obstacles:
                    continue
            valid.append((nx, ny))
    return valid

def cost(current, neighbor):
    if current[0] != neighbor[0] and current[1] != neighbor[1]:
        return math.sqrt(2)
    else:
        return 1

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def a_star_visual(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}
    closed_set = set()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def draw_grid(ax):
        ax.clear()
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.invert_yaxis()
        for ox, oy in obstacles:
            ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black'))

    # fig.text를 이용해 텍스트 고정 (항상 보임)
    text_obj = fig.text(0.1, 0.05, "", fontsize=10)

    path = []
    total_cost = 0

    while open_set:
        _, cost_to_current, current = heapq.heappop(open_set)
        closed_set.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            total_cost = cost_so_far[current]

            draw_grid(ax1)
            ax1.plot([p[0] for p in path], [p[1] for p in path], color='blue', linewidth=2, label='Path (Final)')
            ax1.scatter(*start, c='green', s=100, label='Start')
            ax1.scatter(*goal, c='red', s=100, label='Goal')
            ax1.set_title('Shortest Path (Final)')
            ax1.legend(loc='upper right')

            draw_grid(ax2)
            if open_set:
                open_nodes = [node for _, _, node in open_set]
                ax2.scatter([n[0] for n in open_nodes], [n[1] for n in open_nodes], c='orange', label='Open Set (Candidates)')
            if closed_set:
                ax2.scatter([n[0] for n in closed_set], [n[1] for n in closed_set], c='gray', label='Closed Set (Explored)')
            ax2.scatter(*start, c='green', s=100, label='Start')
            ax2.scatter(*goal, c='red', s=100, label='Goal')
            ax2.set_title('Search Process')
            ax2.legend(loc='upper right')
            ax2.text(0, -1.2, "Orange: Open set\nGray: Closed set\nGreen: Start\nRed: Goal", fontsize=9)

            text_obj.set_text(f"Total distance: {total_cost:.2f} units\nGreen: Start\nRed: Goal")

            fig.canvas.draw()
            fig.canvas.flush_events()
            break

        for next_node in neighbors(current, obstacles):
            new_cost = cost_so_far[current] + cost(current, next_node)
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(open_set, (priority, new_cost, next_node))
                came_from[next_node] = current

        draw_grid(ax1)
        if current in came_from:
            temp_path = reconstruct_path(came_from, current)
            ax1.plot([p[0] for p in temp_path], [p[1] for p in temp_path], color='blue', linewidth=2, label='Path (Current)')
        ax1.scatter(*start, c='green', s=100, label='Start')
        ax1.scatter(*goal, c='red', s=100, label='Goal')
        ax1.set_title('Shortest Path (Current)')
        ax1.legend(loc='upper right')

        if path:
            text_obj.set_text(f"Total distance: {total_cost:.2f} units\nGreen: Start\nRed: Goal")
        else:
            text_obj.set_text("Green: Start\nRed: Goal")

        draw_grid(ax2)
        if open_set:
            open_nodes = [node for _, _, node in open_set]
            ax2.scatter([n[0] for n in open_nodes], [n[1] for n in open_nodes], c='orange', label='Open Set (Candidates)')
        if closed_set:
            ax2.scatter([n[0] for n in closed_set], [n[1] for n in closed_set], c='gray', label='Closed Set (Explored)')
        ax2.scatter(*start, c='green', s=100, label='Start')
        ax2.scatter(*goal, c='red', s=100, label='Goal')
        ax2.set_title('Search Process')
        ax2.legend(loc='upper right')
        ax2.text(0, -1.2, "Orange: Open set\nGray: Closed set\nGreen: Start\nRed: Goal", fontsize=9)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)

    plt.ioff()
    plt.show()
    return path

if __name__ == "__main__":
    obstacles = generate_obstacles()
    print(f"Obstacles: {sorted(obstacles)}")
    path = a_star_visual(start, goal, obstacles)
    if not path:
        print("No path found.")
    else:
        print("Shortest path:", path)

