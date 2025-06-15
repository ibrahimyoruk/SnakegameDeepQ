import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
q_table = np.load("q_table.npy", allow_pickle=True).item()

def get_state(snake, food, direction, grid_size=10):
    head_x, head_y = snake[0]
    food_dx = (food[0] - head_x) // abs(food[0] - head_x) if food[0] != head_x else 0
    food_dy = (food[1] - head_y) // abs(food[1] - head_y) if food[1] != head_y else 0
    danger_straight = 0
    danger_left = 0
    danger_right = 0
    if direction == 'UP':
        if head_y == 0: danger_straight = 1
        if head_x == 0: danger_left = 1
        if head_x == grid_size-1: danger_right = 1
    if direction == 'DOWN':
        if head_y == grid_size-1: danger_straight = 1
        if head_x == grid_size-1: danger_left = 1
        if head_x == 0: danger_right = 1
    if direction == 'LEFT':
        if head_x == 0: danger_straight = 1
        if head_y == grid_size-1: danger_left = 1
        if head_y == 0: danger_right = 1
    if direction == 'RIGHT':
        if head_x == grid_size-1: danger_straight = 1
        if head_y == 0: danger_left = 1
        if head_y == grid_size-1: danger_right = 1
    dir_onehot = [int(direction=='UP'), int(direction=='DOWN'), int(direction=='LEFT'), int(direction=='RIGHT')]
    return (food_dx, food_dy, danger_straight, danger_left, danger_right, *dir_onehot)

def ai_action(snake, food, direction):
    state = get_state(snake, food, direction, grid_size=10)
    state_key = tuple(state)
    if state_key in q_table:
        action_idx = np.argmax(q_table[state_key])
        return ACTIONS[action_idx]
    else:
        return direction
