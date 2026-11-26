"""
Professional broadcast overlay drawer.

Draws skeleton-style player wireframes with joint dots, team-colored speed badges,
foot ellipses, and ball possession indicators on the broadcast video frame —
matching the reference Basketball AI Analytics style.
"""

import cv2
import numpy as np
import sys
sys.path.append('../')
from .utils import draw_rounded_rectangle


class ProOverlayDrawer:
    """
    Draws professional player overlays on the broadcast frame.
    
    Features:
    - Skeleton wireframe rendering on detected players with white joint nodes
    - Team-colored floor ellipses
    - Team-colored speed badges above players
    - Ball possession ring indicator
    """

    # Team colors in BGR format
    TEAM_COLORS = {
        1: (240, 160, 30),    # Bright Cyan-Blue for Team 1 (PHI 76ers)
        2: (20, 190, 255),    # Vibrant Gold/Yellow for Team 2 (IND Pacers)
    }

    JOINT_COLOR = (255, 255, 255)  # White dots for joints

    def __init__(self, team_1_color=None, team_2_color=None,
                 team_1_name="PHI 76ERS", team_2_name="IND PACERS"):
        if team_1_color:
            self.TEAM_COLORS[1] = tuple(team_1_color)
        if team_2_color:
            self.TEAM_COLORS[2] = tuple(team_2_color)
        self.team_1_name = team_1_name
        self.team_2_name = team_2_name

    def draw(self, video_frames, player_tracks, player_assignment,
             ball_acquisition, player_speed_per_frame):
        """
        Draw professional overlays on all frames.

        Args:
            video_frames (list): Input frames.
            player_tracks (list[dict]): Per-frame player tracks.
            player_assignment (list[dict]): Per-frame team assignments.
            ball_acquisition (list[int]): Per-frame ball holder.
            player_speed_per_frame (list[dict]): Per-frame player speeds.

        Returns:
            list: Annotated frames.
        """
        output_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()
            tracks = player_tracks[frame_idx] if frame_idx < len(player_tracks) else {}
            assigns = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            ba = ball_acquisition[frame_idx] if frame_idx < len(ball_acquisition) else -1
            speeds = player_speed_per_frame[frame_idx] if frame_idx < len(player_speed_per_frame) else {}

            for pid, pdata in tracks.items():
                bbox = pdata['bbox']
                team = assigns.get(pid, 1)
                color = self.TEAM_COLORS.get(team, self.TEAM_COLORS[1])
                speed = speeds.get(pid, None)

                # 1. Foot ellipse on floor
                self._draw_foot_ellipse(frame, bbox, color)

                # 2. Skeleton wireframe with joint nodes
                self._draw_wireframe(frame, bbox, color)

                # 3. Speed badge above player
                if speed is not None and speed > 0.5:
                    self._draw_speed_badge(frame, bbox, speed, color)

                # 4. Ball possession indicator
                if pid == ba:
                    self._draw_possession_ring(frame, bbox)

            output_frames.append(frame)

        return output_frames

    def _draw_wireframe(self, frame, bbox, color):
        """Draw an anatomical skeleton wireframe with white joints."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        w = max(10, x2 - x1)
        h = max(20, y2 - y1)

        # Proportions
        head_y = y1 + int(h * 0.08)
        neck_y = y1 + int(h * 0.16)
        shoulder_y = y1 + int(h * 0.22)
        hip_y = y1 + int(h * 0.52)
        knee_y = y1 + int(h * 0.74)
        ankle_y = y2 - int(h * 0.03)

        shoulder_w = int(w * 0.38)
        hip_w = int(w * 0.22)
        knee_w = int(w * 0.20)
        ankle_w = int(w * 0.22)

        thickness = 2
        glow_thickness = 4

        # Key joint positions
        head_center = (cx, head_y)
        neck = (cx, neck_y)
        sh_l = (cx - shoulder_w, shoulder_y)
        sh_r = (cx + shoulder_w, shoulder_y)
        elbow_l = (cx - shoulder_w - int(w * 0.10), shoulder_y + int(h * 0.12))
        elbow_r = (cx + shoulder_w + int(w * 0.10), shoulder_y + int(h * 0.12))
        wrist_l = (cx - shoulder_w - int(w * 0.04), shoulder_y + int(h * 0.24))
        wrist_r = (cx + shoulder_w + int(w * 0.04), shoulder_y + int(h * 0.24))
        hip_center = (cx, hip_y)
        hip_l = (cx - hip_w, hip_y)
        hip_r = (cx + hip_w, hip_y)
        knee_l = (cx - knee_w, knee_y)
        knee_r = (cx + knee_w, knee_y)
        ankle_l = (cx - ankle_w, ankle_y)
        ankle_r = (cx + ankle_w, ankle_y)

        # Draw bones (limbs)
        bones = [
            (neck, hip_center),          # Spine
            (sh_l, sh_r),                # Shoulder bar
            (sh_l, elbow_l), (elbow_l, wrist_l),  # Left arm
            (sh_r, elbow_r), (elbow_r, wrist_r),  # Right arm
            (hip_l, hip_r),              # Pelvis bar
            (hip_l, knee_l), (knee_l, ankle_l),   # Left leg
            (hip_r, knee_r), (knee_r, ankle_r),   # Right leg
        ]

        # Draw glow / stroke
        for pt1, pt2 in bones:
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Head circle
        head_r = max(5, int(h * 0.065))
        cv2.circle(frame, head_center, head_r, color, thickness, cv2.LINE_AA)

        # Draw white circular nodes at each joint
        joints = [
            neck, sh_l, sh_r, elbow_l, elbow_r, wrist_l, wrist_r,
            hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r
        ]
        node_r = 3
        for j in joints:
            cv2.circle(frame, j, node_r, self.JOINT_COLOR, -1, cv2.LINE_AA)

    def _draw_foot_ellipse(self, frame, bbox, color):
        """Draw an ellipse on the court at the player's feet."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        w = max(10, x2 - x1)
        ellipse_w = max(16, int(w * 0.65))
        ellipse_h = max(5, int(w * 0.22))

        # Thin transparent fill
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, int(y2)), (ellipse_w, ellipse_h),
                    0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # Outer ring stroke
        cv2.ellipse(frame, (cx, int(y2)), (ellipse_w, ellipse_h),
                    0, 0, 360, color, 2, cv2.LINE_AA)

    def _draw_speed_badge(self, frame, bbox, speed, color):
        """Draw a rounded speed pill badge above the player's head."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2

        text = f"{int(round(speed))} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        pad_x, pad_y = 6, 3
        badge_w = tw + 2 * pad_x
        badge_h = th + 2 * pad_y

        bx1 = cx - badge_w // 2
        by1 = y1 - badge_h - 7
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h

        # Filled rounded badge in team color
        draw_rounded_rectangle(frame, (bx1, by1), (bx2, by2), color, radius=5, thickness=-1)
        # Subtle dark border
        draw_rounded_rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), radius=5, thickness=1)

        # Crisp black text
        tx = bx1 + pad_x
        ty = by1 + pad_y + th
        cv2.putText(frame, text, (tx, ty), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_possession_ring(self, frame, bbox):
        """Draw a pulsing gold ring around the player with the ball."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        w = max(10, x2 - x1)
        ring_r = max(18, int(w * 0.7))
        ring_y = int(y2)
        cv2.ellipse(frame, (cx, ring_y), (ring_r, int(ring_r * 0.28)),
                    0, 0, 360, (20, 215, 255), 3, cv2.LINE_AA)
