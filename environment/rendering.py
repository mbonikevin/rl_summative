"""Pygame drawing of the Kigali map. Run this file to watch a random agent."""
import numpy as np
import pygame

from environment.custom_env import MAP, NAMES, POSITIONS, SIZE, is_rush_hour

CELL = 64
PANEL = 260
COLORS = {
    ".": (210, 210, 200),  # normal road
    "#": (70, 90, 60),     # hills / buildings
    "B": (235, 190, 120),  # city centre
    "D": (90, 140, 220),   # depot
}
STOP = (150, 200, 150)
DONE = (120, 120, 120)


class Renderer:
    def __init__(self, mode="human"):
        self.mode = mode
        pygame.init()
        size = (SIZE * CELL + PANEL, SIZE * CELL)
        if mode == "human":
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Kigali Freight - truck delivery agent")
        else:
            self.screen = pygame.Surface(size)
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.small = pygame.font.SysFont("dejavusans", 12)
        self.big = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.clock = pygame.time.Clock()

    def draw(self, env):
        if self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
        s = self.screen
        s.fill((30, 30, 35))
        rush = is_rush_hour(env.clock)

        # map cells
        for y, row in enumerate(MAP):
            for x, ch in enumerate(row):
                color = COLORS.get(ch, COLORS["."])
                if ch == "B" and rush:
                    color = (220, 80, 70)  # traffic jam
                if ch in env.deadlines:
                    color = STOP if ch in env.pending else DONE
                pygame.draw.rect(s, color, (x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2))
                if ch in NAMES:
                    s.blit(self.small.render(NAMES[ch], True, (20, 20, 20)), (x * CELL + 3, y * CELL + 3))

        # rain overlay
        if env.raining:
            rain = pygame.Surface((SIZE * CELL, SIZE * CELL), pygame.SRCALPHA)
            rng = np.random.default_rng(env.steps)
            for _ in range(120):
                x, y = rng.integers(0, SIZE * CELL, size=2)
                pygame.draw.line(rain, (120, 160, 255, 160), (x, y), (x - 4, y + 12), 2)
            s.blit(rain, (0, 0))

        # truck
        tx, ty = env.pos
        body = pygame.Rect(tx * CELL + 12, ty * CELL + 22, 40, 24)
        pygame.draw.rect(s, (250, 200, 30), body, border_radius=4)
        pygame.draw.rect(s, (40, 40, 40), (body.right - 12, body.y + 4, 10, 10))
        for wx in (body.x + 8, body.right - 8):
            pygame.draw.circle(s, (20, 20, 20), (wx, body.bottom), 5)

        # side panel
        px = SIZE * CELL + 15
        h, m = divmod(env.clock % 1440, 60)
        lines = [
            (self.big, f"Time {h:02d}:{m:02d}"),
            (self.font, "RUSH HOUR" if rush else "normal traffic"),
            (self.font, "Raining" if env.raining else "Dry"),
            (self.font, ""),
            (self.font, f"Step {env.steps}"),
            (self.font, f"Last reward {env.last_reward:+.1f}"),
            (self.font, f"Total reward {env.total_reward:+.1f}"),
            (self.font, f"On time {env.on_time}  Late {env.late}"),
            (self.font, ""),
            (self.big, "Deliveries"),
        ]
        for c, deadline in env.deadlines.items():
            dh, dm = divmod(deadline % 1440, 60)
            status = "pending" if c in env.pending else "done"
            lines.append((self.font, f"{NAMES[c]} by {dh:02d}:{dm:02d} {status}"))
        for i, (font, text) in enumerate(lines):
            s.blit(font.render(text, True, (235, 235, 235)), (px, 15 + i * 28))

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(6)
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # random agent demo: shows the environment without any training
    from environment.custom_env import KigaliFreightEnv

    env = KigaliFreightEnv(render_mode="human")
    for episode in range(3):
        obs, info = env.reset(seed=episode)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            print(f"step {env.steps:3d}  pos {env.pos}  reward {reward:+.1f}  info {info}")
            done = terminated or truncated
    env.close()
