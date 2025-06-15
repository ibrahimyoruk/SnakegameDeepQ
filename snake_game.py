import pygame
import random
import time
from ai_agent import ai_action

# Oyun ayarları
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 40   # 10x10 grid için!
FPS = 10

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (213, 50, 80)
WHITE = (255, 255, 255)

def select_mode():
    print("Mod Seçimi: ")
    print("1 - Manuel (Klavye ile oyna)")
    print("2 - AI (Yılan kendi oynar)")
    mode = input("Seçiminizi girin (1 veya 2): ")
    if mode == "2":
        return "AI"
    else:
        return "MANUAL"

def draw_snake(snake_blocks, screen):
    for block in snake_blocks:
        pygame.draw.rect(screen, GREEN, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])

def show_score(screen, score, font):
    value = font.render("Skor: " + str(score), True, WHITE)
    screen.blit(value, [10, 10])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")
    font = pygame.font.SysFont("bahnschrift", 25)
    clock = pygame.time.Clock()

    mode = select_mode()

    # 10x10 grid koordinatları
    start_x = 5
    start_y = 5
    snake = [[start_x, start_y]]
    direction = 'RIGHT'
    change_to = direction
    score = 0

    # Yem oluştur
    food_x = random.randint(0, 9)
    food_y = random.randint(0, 9)

    running = True
    while running:
        if mode == "MANUAL":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and direction != 'RIGHT':
                        change_to = 'LEFT'
                    elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                        change_to = 'RIGHT'
                    elif event.key == pygame.K_UP and direction != 'DOWN':
                        change_to = 'UP'
                    elif event.key == pygame.K_DOWN and direction != 'UP':
                        change_to = 'DOWN'
        elif mode == "AI":
            pygame.event.pump()
            change_to = ai_action(snake, [food_x, food_y], direction)
            time.sleep(0.05)

        direction = change_to

        # Hareket
        head_x, head_y = snake[0]
        if direction == 'RIGHT':
            head_x += 1
        elif direction == 'LEFT':
            head_x -= 1
        elif direction == 'UP':
            head_y -= 1
        elif direction == 'DOWN':
            head_y += 1
        new_head = [head_x, head_y]

        # Çarpma
        if (head_x < 0 or head_x > 9 or head_y < 0 or head_y > 9):
            print("Duvara çarptı, oyun bitti! Skor:", score)
            time.sleep(2)
            running = False
            continue
        if new_head in snake:
            print("Kendine çarptı, oyun bitti! Skor:", score)
            time.sleep(2)
            running = False
            continue

        snake.insert(0, new_head)

        # Yem yeme kontrolü
        if head_x == food_x and head_y == food_y:
            score += 1
            while True:
                food_x = random.randint(0, 9)
                food_y = random.randint(0, 9)
                if [food_x, food_y] not in snake:
                    break
        else:
            snake.pop()

        # Çizimler
        screen.fill(BLACK)
        draw_snake([[block[0]*BLOCK_SIZE, block[1]*BLOCK_SIZE] for block in snake], screen)
        pygame.draw.rect(screen, RED, [food_x*BLOCK_SIZE, food_y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
        show_score(screen, score, font)
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
