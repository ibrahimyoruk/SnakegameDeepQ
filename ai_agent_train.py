import numpy as np
import random

GRID_SIZE = 10  # 10x10
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
EPISODES = 10000

def get_state(snake, food, direction):
    head_x, head_y = snake[0]
    food_dx = (food[0] - head_x) // abs(food[0] - head_x) if food[0] != head_x else 0
    food_dy = (food[1] - head_y) // abs(food[1] - head_y) if food[1] != head_y else 0
    danger_straight = 0
    danger_left = 0
    danger_right = 0

    # Basit duvar tehlike kontrolü
    if direction == 'UP':
        if head_y == 0: danger_straight = 1
        if head_x == 0: danger_left = 1
        if head_x == GRID_SIZE-1: danger_right = 1
    if direction == 'DOWN':
        if head_y == GRID_SIZE-1: danger_straight = 1
        if head_x == GRID_SIZE-1: danger_left = 1
        if head_x == 0: danger_right = 1
    if direction == 'LEFT':
        if head_x == 0: danger_straight = 1
        if head_y == GRID_SIZE-1: danger_left = 1
        if head_y == 0: danger_right = 1
    if direction == 'RIGHT':
        if head_x == GRID_SIZE-1: danger_straight = 1
        if head_y == 0: danger_left = 1
        if head_y == GRID_SIZE-1: danger_right = 1

    dir_onehot = [int(direction=='UP'), int(direction=='DOWN'), int(direction=='LEFT'), int(direction=='RIGHT')]
    return (food_dx, food_dy, danger_straight, danger_left, danger_right, *dir_onehot)

def state_to_key(state):
    return tuple(state)

def get_next_position(head, action):
    x, y = head
    if action == 'UP':
        return [x, y-1]
    if action == 'DOWN':
        return [x, y+1]
    if action == 'LEFT':
        return [x-1, y]
    if action == 'RIGHT':
        return [x+1, y]

q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.01
decay = 0.995

for ep in range(EPISODES):
    snake = [[random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]]
    food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
    while food == snake[0]:
        food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
    direction = random.choice(ACTIONS)
    score = 0
    done = False

    while not done:
        state = get_state(snake, food, direction)
        state_key = state_to_key(state)
        if state_key not in q_table:
            q_table[state_key] = np.zeros(len(ACTIONS))
        if random.uniform(0,1) < epsilon:
            action_idx = random.randint(0,3)
        else:
            action_idx = np.argmax(q_table[state_key])
        action = ACTIONS[action_idx]
        next_head = get_next_position(snake[0], action)
        reward = -0.1
        if (next_head[0] < 0 or next_head[0] >= GRID_SIZE or
                next_head[1] < 0 or next_head[1] >= GRID_SIZE):
            reward = -100
            done = True
        elif next_head == food:
            reward = 10
            score += 1
            snake.insert(0, next_head)
            food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            while food in snake:
                food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        elif next_head in snake:
            reward = -100
            done = True
        else:
            snake.insert(0, next_head)
            snake.pop()
        next_state = get_state(snake, food, action)
        next_state_key = state_to_key(next_state)
        if next_state_key not in q_table:
            q_table[next_state_key] = np.zeros(len(ACTIONS))
        q_table[state_key][action_idx] = q_table[state_key][action_idx] + alpha * (
                reward + gamma * np.max(q_table[next_state_key]) - q_table[state_key][action_idx])
        if done:
            print(f"Episode: {ep+1}, Score: {score}, Epsilon: {epsilon:.2f}")
            break
    epsilon = max(min_epsilon, epsilon * decay)

    # ... Diğer kodlar ...
old_dist = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])
next_head = get_next_position(snake[0], action)
new_dist = abs(next_head[0] - food[0]) + abs(next_head[1] - food[1])
reward = -0.1  # Her adımda cezalandır

if new_dist < old_dist:
    reward += 1   # Yaklaşınca daha fazla ödül
else:
    reward -= 1   # Uzaklaşınca daha büyük ceza

# Yem yediyse
if next_head == food:
    reward = 10

# Ölürse
if (next_head[0] < 0 or next_head[0] >= GRID_SIZE or
        next_head[1] < 0 or next_head[1] >= GRID_SIZE or
        next_head in snake):
    reward = -100
    done = True


np.save("q_table.npy", q_table)
print("Q-table kaydedildi!")
