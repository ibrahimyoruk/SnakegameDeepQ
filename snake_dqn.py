import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pygame
from collections import deque

# Oyun parametreleri
GRID_SIZE = 20
BLOCK_SIZE = 20
WIDTH = GRID_SIZE * BLOCK_SIZE
HEIGHT = GRID_SIZE * BLOCK_SIZE
FPS = 10

# DQN parametreleri
BATCH_SIZE = 256
LR = 0.001
GAMMA = 0.9
MEM_SIZE = 100_000
EPISODES = 500
MAX_STEPS = 500

# State = [baş_x, baş_y, food_x, food_y, yön_x, yön_y]
def get_state(snake, food, direction):
    head_x, head_y = snake[0]
    dir_x = direction[0]
    dir_y = direction[1]
    state = np.array([head_x / GRID_SIZE, head_y / GRID_SIZE, food[0] / GRID_SIZE, food[1] / GRID_SIZE, dir_x, dir_y], dtype=np.float32)
    return state

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEM_SIZE)
        self.model = DQN(6, 4)
        self.target = DQN(6, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.update_target()
        self.steps = 0

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qvals = self.model(state)
        return torch.argmax(qvals).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        qvals = self.model(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones) * GAMMA * q_next
        loss = nn.MSELoss()(qvals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.update_target()

# Yön vektörleri ve dönüş
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Sağ, aşağı, sol, yukarı
def turn(direction, action):  # action: 0=devam, 1=sağa, 2=sola, 3=ters
    idx = DIRECTIONS.index(direction)
    if action == 1:
        return DIRECTIONS[(idx + 1) % 4]
    elif action == 2:
        return DIRECTIONS[(idx - 1) % 4]
    elif action == 3:
        return DIRECTIONS[(idx + 2) % 4]
    else:
        return direction

# Oyun ortamı
def play_one_episode(agent, epsilon=0.1, render=False):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    snake = [[GRID_SIZE // 2, GRID_SIZE // 2]]
    direction = random.choice(DIRECTIONS)
    food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
    while food in snake:
        food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
    score = 0
    done = False
    steps = 0

    state = get_state(snake, food, direction)
    while not done and steps < MAX_STEPS:
        steps += 1
        action = agent.act(state, epsilon)
        new_direction = turn(direction, action)
        new_head = [snake[0][0] + new_direction[0], snake[0][1] + new_direction[1]]

        # Çarpma
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                new_head[1] < 0 or new_head[1] >= GRID_SIZE or
                new_head in snake):
            reward = -1
            done = True
        elif new_head == food:
            reward = 1
            score += 1
            snake.insert(0, new_head)
            food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            while food in snake:
                food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        else:
            reward = -0.01  # hayatta kalınca az ceza, yem bulmazsa
            snake.insert(0, new_head)
            snake.pop()

        next_state = get_state(snake, food, new_direction)
        agent.remember(state, action, reward, next_state, done)
        agent.steps += 1
        agent.replay()
        state = next_state
        direction = new_direction

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((0,0,0))
            for block in snake:
                pygame.draw.rect(screen, (0,255,0), [block[0]*BLOCK_SIZE, block[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
            pygame.draw.rect(screen, (255,0,0), [food[0]*BLOCK_SIZE, food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()
    return score

def main():
    agent = Agent()
    best = 0
    for ep in range(EPISODES):
        epsilon = max(0.01, 1 - ep / (EPISODES//2))
        score = play_one_episode(agent, epsilon=epsilon, render=False)
        best = max(best, score)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{EPISODES} | Score: {score} | Best: {best} | Epsilon: {epsilon:.2f}")

    print("Eğitim tamamlandı! Şimdi izleyelim:")
    # Sonunda AI ile oynat ve izle
    for _ in range(5):
        play_one_episode(agent, epsilon=0, render=True)

if __name__ == "__main__":
    main()
