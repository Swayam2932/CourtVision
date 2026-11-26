"""
Professional canvas composer.

Assembles the complete 1080×1350 portrait broadcast analytics frame:
1. Dark header with title, subtitle, and team legend pill
2. Broadcast frame with skeleton overlays and speed badges
3. Section divider with 'LIVE TACTICAL VIEW' label and attacking team pill button
4. Full-court dark-themed tactical view with Voronoi space control
5. 4-card stats dashboard at the bottom
"""

import cv2
import numpy as np
import sys
sys.path.append('../')
from .utils import draw_rounded_rectangle


class ProCanvasComposer:
    """
    Composites all visual layers into the final 1080×1350 portrait canvas.
    """

    CANVAS_WIDTH = 1080
    CANVAS_HEIGHT = 1350

    # Layout dimensions
    HEADER_H = 80
    BROADCAST_H = 600
    DIVIDER_H = 45
    TACTICAL_H = 500
    DASHBOARD_H = 125

    # Colors (BGR)
    CANVAS_BG = (16, 13, 8)
    HEADER_BG = (16, 13, 8)
    DIVIDER_BG = (20, 16, 11)
    PILL_BG = (35, 27, 20)
    PILL_BORDER = (55, 45, 32)
    TITLE_COLOR = (255, 255, 255)
    SUBTITLE_COLOR = (140, 135, 130)

    TEAM1_COLOR = (240, 160, 30)  # Cyan-blue (PHI)
    TEAM2_COLOR = (20, 190, 255)  # Amber-gold (IND)

    def __init__(self, team_1_name="PHI 76ERS", team_2_name="IND PACERS"):
        self.team_1_name = team_1_name
        self.team_2_name = team_2_name

    def compose(self, broadcast_frame, tactical_view, dashboard_bar,
                attacking_team=2, attacking_direction='left'):
        """
        Assemble the final canvas.

        Args:
            broadcast_frame (np.ndarray): Video frame with overlays.
            tactical_view (np.ndarray): Full-court tactical view.
            dashboard_bar (np.ndarray): 4-card bottom stats dashboard.
            attacking_team (int): 1 or 2.
            attacking_direction (str): 'left' or 'right'.

        Returns:
            np.ndarray: 1080x1350 portrait canvas.
        """
        canvas = np.full((self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 3), self.CANVAS_BG, dtype=np.uint8)

        # 1. Header (y: 0 -> 80)
        self._draw_header(canvas)

        # 2. Broadcast Frame (y: 80 -> 680)
        broadcast_fitted = self._fit_broadcast(broadcast_frame, self.CANVAS_WIDTH, self.BROADCAST_H)
        canvas[self.HEADER_H:self.HEADER_H + self.BROADCAST_H, :] = broadcast_fitted

        # 3. Divider (y: 680 -> 725)
        divider_y = self.HEADER_H + self.BROADCAST_H
        self._draw_divider(canvas, divider_y, attacking_team, attacking_direction)

        # 4. Tactical View (y: 725 -> 1225)
        tactical_y = divider_y + self.DIVIDER_H
        tactical_fitted = cv2.resize(tactical_view, (self.CANVAS_WIDTH, self.TACTICAL_H))
        canvas[tactical_y:tactical_y + self.TACTICAL_H, :] = tactical_fitted

        # 5. Dashboard (y: 1225 -> 1350)
        dashboard_y = tactical_y + self.TACTICAL_H
        dashboard_fitted = cv2.resize(dashboard_bar, (self.CANVAS_WIDTH, self.DASHBOARD_H))
        canvas[dashboard_y:self.CANVAS_HEIGHT, :] = dashboard_fitted

        return canvas

    def _draw_header(self, canvas):
        """Draw top header with title and team pill legend."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(canvas, "BASKETBALL AI ANALYTICS", (28, 42),
                    font, 0.95, self.TITLE_COLOR, 2, cv2.LINE_AA)

        # Subtitle
        cv2.putText(canvas, "Player tracking . Team ID . Camera calibration . Space control",
                    (28, 64), font, 0.38, self.SUBTITLE_COLOR, 1, cv2.LINE_AA)

        # Team Legend Pill on top right
        pill_w = 260
        pill_h = 32
        px2 = self.CANVAS_WIDTH - 24
        px1 = px2 - pill_w
        py1 = 24
        py2 = py1 + pill_h

        draw_rounded_rectangle(canvas, (px1, py1), (px2, py2), self.PILL_BG, radius=16, thickness=-1)
        draw_rounded_rectangle(canvas, (px1, py1), (px2, py2), self.PILL_BORDER, radius=16, thickness=1)

        # Team 1 (Blue)
        cv2.circle(canvas, (px1 + 18, py1 + 16), 5, self.TEAM1_COLOR, -1, cv2.LINE_AA)
        cv2.putText(canvas, self.team_1_name, (px1 + 30, py1 + 21),
                    font, 0.40, self.TITLE_COLOR, 1, cv2.LINE_AA)

        # Team 2 (Gold)
        cv2.circle(canvas, (px1 + 138, py1 + 16), 5, self.TEAM2_COLOR, -1, cv2.LINE_AA)
        cv2.putText(canvas, self.team_2_name, (px1 + 150, py1 + 21),
                    font, 0.40, self.TITLE_COLOR, 1, cv2.LINE_AA)

    def _draw_divider(self, canvas, y, attacking_team, attacking_direction):
        """Draw divider with title and live attacking team pill button."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Background fill
        cv2.rectangle(canvas, (0, y), (self.CANVAS_WIDTH, y + self.DIVIDER_H), self.DIVIDER_BG, -1)

        # Left label: LIVE TACTICAL VIEW
        cv2.putText(canvas, "LIVE TACTICAL VIEW", (28, y + 28),
                    font, 0.46, self.TITLE_COLOR, 2, cv2.LINE_AA)

        cv2.putText(canvas, "court homography . Voronoi space control", (210, y + 28),
                    font, 0.36, self.SUBTITLE_COLOR, 1, cv2.LINE_AA)

        # Right Attacking Pill
        atk_name = self.team_1_name if attacking_team == 1 else self.team_2_name
        # Short team code if long
        if " " in atk_name:
            code = atk_name.split()[0]
        else:
            code = atk_name

        arrow_char = "<-" if attacking_direction == 'left' else "->"
        text = f"{code} ATTACKING {arrow_char}"

        pill_color = self.TEAM1_COLOR if attacking_team == 1 else self.TEAM2_COLOR
        (tw, th), _ = cv2.getTextSize(text, font, 0.48, 2)

        pill_w = tw + 24
        pill_h = 28
        bx2 = self.CANVAS_WIDTH - 24
        bx1 = bx2 - pill_w
        by1 = y + 8
        by2 = by1 + pill_h

        draw_rounded_rectangle(canvas, (bx1, by1), (bx2, by2), pill_color, radius=14, thickness=-1)
        cv2.putText(canvas, text, (bx1 + 12, by1 + 19), font, 0.48, (0, 0, 0), 2, cv2.LINE_AA)

    @staticmethod
    def _fit_broadcast(image, target_w, target_h):
        """Fit broadcast frame to target dimensions without distorting aspect ratio."""
        h, w = image.shape[:2]
        scale = target_w / w
        new_h = int(h * scale)
        resized = cv2.resize(image, (target_w, new_h))

        if new_h >= target_h:
            # Crop center vertically
            start_y = (new_h - target_h) // 2
            return resized[start_y:start_y + target_h, :]
        else:
            # Pad top and bottom
            pad_total = target_h - new_h
            pad_top = pad_total // 2
            pad_bot = pad_total - pad_top
            return cv2.copyMakeBorder(resized, pad_top, pad_bot, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(16, 13, 8))
