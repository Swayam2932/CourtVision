"""
Stats dashboard drawer.

Draws the 4-card bottom analytics dashboard matching broadcast graphics:
1. SPACE CONTROL · FRONTCOURT (team % with dual progress bar)
2. TEAM SPACING · HULL m² (convex hull area with split bar)
3. DISTANCE COVERED · m (cumulative meters per team with split bar)
4. OPEN SPACE · OFFENSE (average gap to nearest defender in meters)
"""

import cv2
import numpy as np
import sys
sys.path.append('../')
from .utils import draw_rounded_rectangle


class StatsDashboardDrawer:
    """
    Draws a 4-card analytics dashboard.
    """

    BG_COLOR = (16, 13, 8)            # Screen background
    CARD_BG = (27, 23, 18)            # Dark card background
    CARD_BORDER = (45, 38, 30)        # Card border stroke
    LABEL_COLOR = (150, 145, 140)     # Dimmed label text
    VALUE_COLOR = (255, 255, 255)     # Default value color
    SUBTITLE_COLOR = (130, 125, 120)  # Subtitle caption color

    # Team metric colors (BGR)
    TEAM1_COLOR = (240, 160, 30)      # Cyan-blue (PHI)
    TEAM2_COLOR = (20, 190, 255)      # Amber-gold (IND)
    BAR_EMPTY = (55, 48, 38)          # Unfilled bar color

    def __init__(self, width=1080, height=125):
        self.width = width
        self.height = height

    def draw(self, frontcourt_pct, team_spacing, cum_distance_t1, cum_distance_t2,
             open_space_m, attacking_team=2, team_1_name="PHI", team_2_name="IND"):
        """
        Draw the 4-card dashboard bar.

        Args:
            frontcourt_pct (dict): {'team1_pct': float, 'team2_pct': float}
            team_spacing (dict): {'team1_hull_m2': float, 'team2_hull_m2': float}
            cum_distance_t1 (float): Cumulative distance team 1 (m).
            cum_distance_t2 (float): Cumulative distance team 2 (m).
            open_space_m (float): Open space in meters.
            attacking_team (int): 1 or 2.
            team_1_name (str): Short name for team 1.
            team_2_name (str): Short name for team 2.

        Returns:
            np.ndarray: Dashboard image (height x width x 3).
        """
        bar = np.full((self.height, self.width, 3), self.BG_COLOR, dtype=np.uint8)

        margin_x = 16
        spacing = 14
        margin_y = 10
        card_h = self.height - 2 * margin_y
        card_w = (self.width - 2 * margin_x - 3 * spacing) // 4

        # Card 1: Space Control
        x1 = margin_x
        self._draw_card_space_control(bar, (x1, margin_y, card_w, card_h), frontcourt_pct)

        # Card 2: Team Spacing
        x2 = x1 + card_w + spacing
        self._draw_card_team_spacing(bar, (x2, margin_y, card_w, card_h), team_spacing)

        # Card 3: Distance Covered
        x3 = x2 + card_w + spacing
        self._draw_card_distance(bar, (x3, margin_y, card_w, card_h), cum_distance_t1, cum_distance_t2)

        # Card 4: Open Space
        x4 = x3 + card_w + spacing
        self._draw_card_open_space(bar, (x4, margin_y, card_w, card_h), open_space_m,
                                  attacking_team, team_1_name, team_2_name)

        return bar

    def _draw_card_base(self, image, rect):
        """Draw base card background with rounded corners."""
        x, y, w, h = rect
        draw_rounded_rectangle(image, (x, y), (x + w, y + h), self.CARD_BG, radius=8, thickness=-1)
        draw_rounded_rectangle(image, (x, y), (x + w, y + h), self.CARD_BORDER, radius=8, thickness=1)

    def _draw_split_bar(self, image, bx, by, bw, bh, ratio):
        """Draw a dual-colored horizontal progress bar with rounded ends."""
        ratio = max(0.02, min(0.98, ratio))
        w1 = int(bw * ratio)

        # Draw left bar (Team 1 Blue)
        draw_rounded_rectangle(image, (bx, by), (bx + bw, by + bh), self.TEAM2_COLOR, radius=3, thickness=-1)
        draw_rounded_rectangle(image, (bx, by), (bx + w1, by + bh), self.TEAM1_COLOR, radius=3, thickness=-1)

    def _draw_card_space_control(self, image, rect, pct):
        self._draw_card_base(image, rect)
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label
        cv2.putText(image, "SPACE CONTROL . FRONTCOURT", (x + 14, y + 24),
                    font, 0.36, self.LABEL_COLOR, 1, cv2.LINE_AA)

        # Values
        t1 = int(round(pct.get('team1_pct', 50)))
        t2 = int(round(pct.get('team2_pct', 50)))

        cv2.putText(image, f"{t1}%", (x + 14, y + 62),
                    font, 0.85, self.TEAM1_COLOR, 2, cv2.LINE_AA)

        t2_str = f"{t2}%"
        (tw, _), _ = cv2.getTextSize(t2_str, font, 0.85, 2)
        cv2.putText(image, t2_str, (x + w - 14 - tw, y + 62),
                    font, 0.85, self.TEAM2_COLOR, 2, cv2.LINE_AA)

        # Split progress bar
        self._draw_split_bar(image, x + 14, y + 78, w - 28, 6, t1 / 100.0)

    def _draw_card_team_spacing(self, image, rect, spacing):
        self._draw_card_base(image, rect)
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label
        cv2.putText(image, "TEAM SPACING . HULL m2", (x + 14, y + 24),
                    font, 0.36, self.LABEL_COLOR, 1, cv2.LINE_AA)

        # Values
        t1 = int(round(spacing.get('team1_hull_m2', 0)))
        t2 = int(round(spacing.get('team2_hull_m2', 0)))

        cv2.putText(image, str(t1), (x + 14, y + 62),
                    font, 0.85, self.TEAM1_COLOR, 2, cv2.LINE_AA)

        t2_str = str(t2)
        (tw, _), _ = cv2.getTextSize(t2_str, font, 0.85, 2)
        cv2.putText(image, t2_str, (x + w - 14 - tw, y + 62),
                    font, 0.85, self.TEAM2_COLOR, 2, cv2.LINE_AA)

        total = t1 + t2
        ratio = (t1 / total) if total > 0 else 0.5
        self._draw_split_bar(image, x + 14, y + 78, w - 28, 6, ratio)

    def _draw_card_distance(self, image, rect, d1, d2):
        self._draw_card_base(image, rect)
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label
        cv2.putText(image, "DISTANCE COVERED . m", (x + 14, y + 24),
                    font, 0.36, self.LABEL_COLOR, 1, cv2.LINE_AA)

        val1 = int(round(d1))
        val2 = int(round(d2))

        cv2.putText(image, str(val1), (x + 14, y + 62),
                    font, 0.85, self.TEAM1_COLOR, 2, cv2.LINE_AA)

        val2_str = str(val2)
        (tw, _), _ = cv2.getTextSize(val2_str, font, 0.85, 2)
        cv2.putText(image, val2_str, (x + w - 14 - tw, y + 62),
                    font, 0.85, self.TEAM2_COLOR, 2, cv2.LINE_AA)

        total = val1 + val2
        if total == 0:
            # Single continuous bar
            draw_rounded_rectangle(image, (x + 14, y + 78), (x + w - 14, y + 84),
                                  self.TEAM2_COLOR, radius=3, thickness=-1)
        else:
            self._draw_split_bar(image, x + 14, y + 78, w - 28, 6, val1 / total)

    def _draw_card_open_space(self, image, rect, open_space, attacking_team, t1_name, t2_name):
        self._draw_card_base(image, rect)
        x, y, w, h = rect
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label
        cv2.putText(image, "OPEN SPACE . OFFENSE", (x + 14, y + 24),
                    font, 0.36, self.LABEL_COLOR, 1, cv2.LINE_AA)

        # Big Value in Attacking Team Color
        val_color = self.TEAM1_COLOR if attacking_team == 1 else self.TEAM2_COLOR
        val_str = f"{open_space:.1f} m"
        cv2.putText(image, val_str, (x + 14, y + 64),
                    font, 0.90, val_color, 2, cv2.LINE_AA)

        # Subtitle caption
        atk_name = t1_name if attacking_team == 1 else t2_name
        caption = f"avg gap {atk_name} -> nearest defender"
        cv2.putText(image, caption, (x + 14, y + 86),
                    font, 0.30, self.SUBTITLE_COLOR, 1, cv2.LINE_AA)
