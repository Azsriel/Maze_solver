"""
Q-learning Maze Solver with Pygame

Features:
- Grid-based maze editable in "Edit Mode":
  * Left-click to toggle walls
  * Right-click to set Start and End (right-click once sets Start, right-click twice sets End)
  * Press 'C' to clear walls
- Buttons:
  * Train: starts Q-learning training and visualizes every episode on the pygame screen
  * Display: after training, animates the learned greedy path from Start to End
- Q-learning parameters configurable in the top area

Run: python q_learning_maze_solver.py
Requires: pygame, numpy

This file is a single-file application. Modify GRID_ROWS/GRID_COLS or cell size to change maze size.
"""

import pygame
import sys
import numpy as np
import random
import time
from collections import defaultdict

# ----------------- Config -----------------
CELL_SIZE = 32
GRID_ROWS = 15
GRID_COLS = 20
TOP_BAR_HEIGHT = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE + 500
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + TOP_BAR_HEIGHT
FPS = 60

# Q-learning hyperparameters (tweakable in-code; UI sliders not provided, simpler)
ALPHA = 0.7          # learning rate
GAMMA = 0.98         # discount factor
EPSILON_START = 1.0  # initial exploration probability
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
EPISODES = 400
MAX_STEPS_PER_EPISODE = GRID_ROWS * GRID_COLS * 4

# Visualization speed
STEP_DELAY = 0.0002   # seconds between agent steps during training display
DISPLAY_DELAY = 0.05 # seconds when showing final learned path

SKIP_ANIM = False

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY  = (200,200,200)
LIGHT_GRAY = (230,230,230)
BLUE  = (66,133,244)
GREEN = (15,157,88)
RED   = (219,68,55)
YELLOW = (244,180,0)

# Actions: Up, Down, Left, Right
ACTIONS = [( -1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['UP','DOWN','LEFT','RIGHT']

# ----------------- Helper functions -----------------

def state_to_index(r, c):
    return r * GRID_COLS + c


def index_to_state(idx):
    return divmod(idx, GRID_COLS)


# ----------------- Agent / Q-Learning -----------------
class QAgent:
    def __init__(self, rows, cols, actions):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.actions = actions
        self.n_actions = len(actions)
        # Q-table as numpy array
        self.Q = np.zeros((self.n_states, self.n_actions))

    def reset_Q(self):
        self.Q[:] = 0.0

    def choose_action(self, state_idx, epsilon, valid_actions_mask):
        # valid_actions_mask: boolean array length n_actions, True if allowed
        if random.random() < epsilon:
            # explore among valid
            valid_indices = [i for i,ok in enumerate(valid_actions_mask) if ok]
            return random.choice(valid_indices)
        else:
            # exploit: choose action with max Q among valid
            qvals = self.Q[state_idx].copy()
            # mask invalid with very low values
            for i, ok in enumerate(valid_actions_mask):
                if not ok:
                    qvals[i] = -1e9
            # argmax tie-breaker random
            maxv = qvals.max()
            candidates = [i for i, v in enumerate(qvals) if abs(v - maxv) < 1e-9]
            return random.choice(candidates)

    def update(self, s_idx, a, r, s2_idx, alpha, gamma):
        best_next = self.Q[s2_idx].max()
        self.Q[s_idx, a] = (1 - alpha) * self.Q[s_idx, a] + alpha * (r + gamma * best_next)

    def greedy_action(self, s_idx, valid_actions_mask):
        qvals = self.Q[s_idx].copy()
        for i, ok in enumerate(valid_actions_mask):
            if not ok:
                qvals[i] = -1e9
        maxv = qvals.max()
        candidates = [i for i, v in enumerate(qvals) if abs(v - maxv) < 1e-9]
        return random.choice(candidates)


# ----------------- Pygame UI -----------------
class Button:
    def __init__(self, rect, text, font, bg=GRAY):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg
        self.hover = False

    def draw(self, surf):
        color = tuple(min(255, c + 10) for c in self.bg) if self.hover else self.bg
        pygame.draw.rect(surf, color, self.rect)
        pygame.draw.rect(surf, BLACK, self.rect, 2)
        txt = self.font.render(self.text, True, BLACK)
        tx = self.rect.x + (self.rect.width - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.height - txt.get_height()) // 2
        surf.blit(txt, (tx, ty))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                return True
        return False


class MazeApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Q-learning Maze Solver')
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 14)

        # grid data
        self.walls = [[False]*GRID_COLS for _ in range(GRID_ROWS)]
        self.start = (0, 0)
        self.end = (GRID_ROWS-1, GRID_COLS-1)
        self.edit_mode = True
        self.placing_start = False
        self.placing_end = False

        # agent & Q
        self.agent = QAgent(GRID_ROWS, GRID_COLS, ACTIONS)
        self.trained = False

        # UI
        self.train_btn = Button((10, 10, 100, 40), 'Train', self.font, bg=BLUE)
        self.display_btn = Button((120, 10, 120, 40), 'Display Path', self.font, bg=GREEN)
        self.clear_btn = Button((250, 10, 80, 40), 'Clear', self.font, bg=YELLOW)
        self.reset_q_btn = Button((340, 10, 120, 40), 'Reset Q', self.font, bg=LIGHT_GRAY)
        self.skip_display = Button((550, 40, 120, 40), 'SKIP', self.font, bg=LIGHT_GRAY)


        # training state
        self.training = False
        self.current_episode = 0
        self.current_step = 0
        self.epsilon = EPSILON_START

        # For visualize an episode
        self.episode_path = []  # list of (r,c) visited in current episode

    def cell_at_pixel(self, pos):
        x, y = pos
        if y < TOP_BAR_HEIGHT:
            return None
        gx = x // CELL_SIZE
        gy = (y - TOP_BAR_HEIGHT) // CELL_SIZE
        if 0 <= gy < GRID_ROWS and 0 <= gx < GRID_COLS:
            return (gy, gx)
        return None

    def draw(self):
        self.screen.fill(WHITE)
        # draw top bar
        pygame.draw.rect(self.screen, LIGHT_GRAY, (0,0,WINDOW_WIDTH, TOP_BAR_HEIGHT))
        # draw buttons
        self.train_btn.draw(self.screen)
        self.display_btn.draw(self.screen)
        self.clear_btn.draw(self.screen)
        self.reset_q_btn.draw(self.screen)
        self.skip_display.draw(self.screen)

        # display text info
        status = 'EDIT MODE' if self.edit_mode else ('TRAINING' if self.training else 'RUNNING')
        info = f'Status: {status} | Episode: {self.current_episode}/{EPISODES} | Step: {self.current_step} | Epsilon: {self.epsilon:.3f}'
        self.screen.blit(self.small_font.render(info, True, BLACK), (480, 22))
        controls = 'Left-click: toggle wall | Right-click: set Start/End | C: clear walls | T: toggle edit mode'
        self.screen.blit(self.small_font.render(controls, True, BLACK), (10, 55))

        # draw grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rect = pygame.Rect(c*CELL_SIZE, TOP_BAR_HEIGHT + r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = WHITE
                if self.walls[r][c]:
                    color = BLACK
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # draw start/end
        sr, sc = self.start
        er, ec = self.end
        start_rect = pygame.Rect(sc*CELL_SIZE+4, TOP_BAR_HEIGHT + sr*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8)
        end_rect = pygame.Rect(ec*CELL_SIZE+4, TOP_BAR_HEIGHT + er*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8)
        pygame.draw.rect(self.screen, GREEN, start_rect)
        pygame.draw.rect(self.screen, RED, end_rect)

        # draw current episode path
        if self.episode_path:
            for pos in self.episode_path:
                r,c = pos
                rect = pygame.Rect(c*CELL_SIZE+8, TOP_BAR_HEIGHT + r*CELL_SIZE+8, CELL_SIZE-16, CELL_SIZE-16)
                pygame.draw.rect(self.screen, (100,100,255), rect)

        pygame.display.flip()

    def neighbors_and_mask(self, r, c):
        # returns list of (nr,nc) for each action and mask of valid actions
        results = []
        mask = []
        for dr,dc in ACTIONS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and not self.walls[nr][nc]:
                results.append((nr,nc))
                mask.append(True)
            else:
                results.append((r,c))  # if invalid, stay in place
                mask.append(False)
        return results, mask

    def train(self):
        # Run Q-learning with visualization per episode
        self.training = True
        self.trained = False
        self.agent.reset_Q()
        self.current_episode = 0
        self.current_step = 0
        self.epsilon = EPSILON_START
        SKIP_ANIM = False

        for ep in range(1, EPISODES+1):
            self.current_episode = ep
            state = self.start
            s_idx = state_to_index(*state)
            self.episode_path = [state]
            step = 0
            done = False

            for step in range(MAX_STEPS_PER_EPISODE):
                self.current_step = step+1
                # handle events so window stays responsive
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            pygame.quit(); sys.exit()
                    if self.skip_display.handle_event(ev):
                        SKIP_ANIM = True
                # choose action
                _, mask = self.neighbors_and_mask(*state)
                a = self.agent.choose_action(s_idx, self.epsilon, mask)
                next_state_candidates, _ = self.neighbors_and_mask(*state)
                next_state = next_state_candidates[a]
                s2_idx = state_to_index(*next_state)

                # reward: high positive for reaching goal, small negative per step
                if next_state == self.end:
                    reward = 10.0
                    self.agent.update(s_idx, a, reward, s2_idx, ALPHA, GAMMA)
                    self.episode_path.append(next_state)
                    done = True
                    # visualize final step of episode
                    if not SKIP_ANIM:
                        self.draw(); time.sleep(STEP_DELAY)
                    break
                else:
                    reward = -0.04
                    # small extra penalty for staying in place due to invalid move
                    if next_state == state and not mask[a]:
                        reward -= 0.2
                    self.agent.update(s_idx, a, reward, s2_idx, ALPHA, GAMMA)

                state = next_state
                s_idx = s2_idx
                self.episode_path.append(state)

                # draw
                if not SKIP_ANIM:
                    self.draw(); time.sleep(STEP_DELAY)

            # decay epsilon
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            # short pause between episodes
            time.sleep(0.01)

        self.training = False
        self.trained = True
        self.current_episode = EPISODES
        self.current_step = 0

    def run_greedy(self):
        # Animate learned greedy path
        if not self.trained:
            print('Not trained yet')
            return
        state = self.start
        path = [state]
        for _ in range(MAX_STEPS_PER_EPISODE):
            s_idx = state_to_index(*state)
            _, mask = self.neighbors_and_mask(*state)
            a = self.agent.greedy_action(s_idx, mask)
            next_state_candidates, _ = self.neighbors_and_mask(*state)
            next_state = next_state_candidates[a]
            path.append(next_state)
            state = next_state
            if state == self.end:
                break
        # animate
        for pos in path:
            self.episode_path = [pos]
            self.draw()
            time.sleep(DISPLAY_DELAY)
        # final highlight of whole path
        self.episode_path = path
        for _ in range(6):
            self.draw(); time.sleep(0.15)

    def mainloop(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    # button clicks
                    if self.train_btn.handle_event(ev):
                        # kick off training loop
                        # run in main thread (blocking) but window remains responsive because we pump events inside train()
                        self.edit_mode = False
                        SKIP = False
                        self.train()
                        SKIP = False
                    elif self.display_btn.handle_event(ev):
                        # display learned path
                        self.edit_mode = False
                        self.run_greedy()
                    elif self.clear_btn.handle_event(ev):
                        # clear walls
                        self.walls = [[False]*GRID_COLS for _ in range(GRID_ROWS)]
                        self.trained = False
                    elif self.reset_q_btn.handle_event(ev):
                        self.agent.reset_Q()
                        self.trained = False
                    else:
                        # grid interactions
                        pos = ev.pos
                        cell = self.cell_at_pixel(pos)
                        if cell and self.edit_mode:
                            r,c = cell
                            if ev.button == 1:  # left click toggle wall
                                # prevent toggling start/end cells into wall
                                if (r,c) != self.start and (r,c) != self.end:
                                    self.walls[r][c] = not self.walls[r][c]
                            elif ev.button == 3:  # right click: set start or end
                                # if same as start, set end next
                                if (r,c) == self.start:
                                    # do nothing
                                    pass
                                elif (r,c) == self.end:
                                    pass
                                else:
                                    # decide whether to set start or end based on whether start is default or user indicated
                                    # we toggle: first right-click sets start, second sets end
                                    # simple heuristic: if clicked cell closer to top-left than end, set start else set end
                                    # but simpler: if not set recently: set start, then set end next time
                                    # We'll set start if currently at default or if user holds Shift
                                    mods = pygame.key.get_mods()
                                    if mods & pygame.KMOD_SHIFT:
                                        self.end = (r,c)
                                    else:
                                        self.start = (r,c)

                if ev.type == pygame.MOUSEMOTION:
                    # hover updates for buttons
                    self.train_btn.handle_event(ev)
                    self.display_btn.handle_event(ev)
                    self.clear_btn.handle_event(ev)
                    self.reset_q_btn.handle_event(ev)
                    self.skip_display.handle_event(ev)

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_c:
                        self.walls = [[False]*GRID_COLS for _ in range(GRID_ROWS)]
                        self.trained = False
                    if ev.key == pygame.K_t:
                        self.edit_mode = not self.edit_mode

            self.draw()
            self.clock.tick(FPS)


if __name__ == '__main__':
    app = MazeApp()
    app.mainloop()
