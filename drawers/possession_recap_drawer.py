"""
Possession recap drawer.

Draws the end-of-possession summary screen overlay matching broadcast graphics:
- Dark semi-transparent background overlay
- Large 'POSSESSION RECAP' title
- Subtitle with possession count and tracked seconds
- 4 key metrics:
  1. DISTANCE COVERED (Team 1 m · Team 2 m)
  2. TOP SPEED (X.X km/h with player/team indicator)
  3. FRONTCOURT SPACE ON OFFENSE (Team 1 % · Team 2 %)
  4. AVG OPEN SPACE ON OFFENSE (Team 1 m · Team 2 m)
"""

import cv2
import numpy as np


class PossessionRecapDrawer:
    """
    Renders a possession recap summary overlay.
    """

    OVERLAY_ALPHA = 0.72
    TITLE_COLOR = (255, 255, 255)
    SUBTITLE_COLOR = (180, 180, 180)
    LABEL_COLOR = (145, 140, 135)
    VALUE_COLOR = (255, 255, 255)

    def __init__(self, team_1_name="PHI", team_2_name="IND"):
        self.team_1_name = team_1_name
        self.team_2_name = team_2_name

    def draw(self, frame, recap):
        """
        Draw possession recap on a frame.

        Args:
            frame (np.ndarray): Video frame.
            recap (dict): Metrics dictionary from PossessionTracker.

        Returns:
            np.ndarray: Frame with recap overlay.
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Dark overlay
        dark = np.zeros_like(frame)
        cv2.addWeighted(dark, self.OVERLAY_ALPHA, frame, 1.0 - self.OVERLAY_ALPHA, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cx = w // 2

        # 1. Main Title
        title = "POSSESSION RECAP"
        (tw, _), _ = cv2.getTextSize(title, font, 1.35, 3)
        cv2.putText(frame, title, (cx - tw // 2, int(h * 0.18)),
                    font, 1.35, self.TITLE_COLOR, 3, cv2.LINE_AA)

        # 2. Subtitle
        sub = f"{recap['num_possessions']} possessions . {recap['duration_seconds']} s tracked"
        (sw, _), _ = cv2.getTextSize(sub, font, 0.48, 1)
        cv2.putText(frame, sub, (cx - sw // 2, int(h * 0.25)),
                    font, 0.48, self.SUBTITLE_COLOR, 1, cv2.LINE_AA)

        # 3. Metrics
        top_team_name = self.team_1_name if recap.get('top_speed_team', 1) == 1 else self.team_2_name
        t1_dist = int(recap.get('team1_distance_m', 0))
        t2_dist = int(recap.get('team2_distance_m', 0))
        top_spd = recap.get('top_speed_kmh', 0.0)
        t1_fc = int(recap.get('team1_frontcourt_pct', 50))
        t2_fc = int(recap.get('team2_frontcourt_pct', 50))
        t1_os = recap.get('team1_avg_open_space', 0.0)
        t2_os = recap.get('team2_avg_open_space', 0.0)

        metrics = [
            ("DISTANCE COVERED",
             f"{self.team_1_name} {t1_dist} m  .  {self.team_2_name} {t2_dist} m"),
            ("TOP SPEED",
             f"{top_spd:.1f} km/h ({top_team_name})"),
            ("FRONTCOURT SPACE ON OFFENSE",
             f"{self.team_1_name} {t1_fc}%  .  {self.team_2_name} {t2_fc}%"),
            ("AVG OPEN SPACE ON OFFENSE",
             f"{self.team_1_name} {t1_os:.1f} m  .  {self.team_2_name} {t2_os:.1f} m"),
        ]

        y_start = int(h * 0.36)
        y_step = int(h * 0.14)

        for i, (label, val_str) in enumerate(metrics):
            y = y_start + i * y_step

            # Label
            (lw, _), _ = cv2.getTextSize(label, font, 0.38, 1)
            cv2.putText(frame, label, (cx - lw // 2, y),
                        font, 0.38, self.LABEL_COLOR, 1, cv2.LINE_AA)

            # Value
            (vw, _), _ = cv2.getTextSize(val_str, font, 0.72, 2)
            cv2.putText(frame, val_str, (cx - vw // 2, y + 30),
                        font, 0.72, self.VALUE_COLOR, 2, cv2.LINE_AA)

        return frame
