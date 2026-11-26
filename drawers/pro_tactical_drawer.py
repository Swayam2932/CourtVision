"""
Professional tactical view drawer with Voronoi space control.

Draws a full-court dark-themed tactical view:
- Dark navy background
- Voronoi space control partition (slate-blue vs olive-gold)
- Court boundary and regulation markings (center circle, 3pt arcs, paint/keys, restricted areas)
- Reference keypoint markers
- Team player position dots with crisp white borders
"""

import cv2
import numpy as np
import sys
sys.path.append('../')
from .utils import draw_rounded_rectangle


class ProTacticalDrawer:
    """
    Renders a full-court dark-theme tactical view matching broadcast graphics.
    """

    # Background & line colors (BGR)
    BG_COLOR = (24, 20, 14)          # Very dark navy / charcoal
    COURT_FILL = (30, 25, 18)        # Neutral court area fill
    COURT_LINE_COLOR = (75, 70, 65)  # Crisp muted court line markings
    REF_MARKER = (90, 85, 75)        # Subtle reference markers

    # Team Voronoi space control colors (BGR)
    TEAM1_VORONOI = (79, 67, 32)     # Deep slate-blue (PHI 76ers)
    TEAM2_VORONOI = (45, 70, 85)     # Warm olive-gold (IND Pacers)

    # Player dot colors (BGR)
    TEAM1_DOT = (240, 160, 30)       # Cyan-blue fill
    TEAM2_DOT = (20, 190, 255)       # Gold/yellow fill
    DOT_BORDER = (255, 255, 255)     # Crisp white border

    def __init__(self, court_width=1000, court_height=480):
        self.width = court_width
        self.height = court_height
        self.court_m_w = 28.0
        self.court_m_h = 15.0

    def draw_tactical_frame(self, tactical_positions, player_assignment_frame,
                             ball_acquisition_frame, voronoi_regions,
                             attacking_team=1, attacking_direction='left',
                             team_1_name="PHI 76ERS", team_2_name="IND PACERS"):
        """
        Draw a single tactical view frame.

        Args:
            tactical_positions (dict): {player_id: (x, y)} in tactical coords (300x161).
            player_assignment_frame (dict): {player_id: team_id}.
            ball_acquisition_frame (int): Player with ball, or -1.
            voronoi_regions (list): Output from VoronoiSpaceControl.
            attacking_team (int): 1 or 2.
            attacking_direction (str): 'left' or 'right'.

        Returns:
            np.ndarray: Tactical view frame (height x width x 3).
        """
        canvas = np.full((self.height, self.width, 3), self.BG_COLOR, dtype=np.uint8)

        # Court bounds with padding
        pad_x = 24
        pad_y = 18
        cw = self.width - 2 * pad_x
        ch = self.height - 2 * pad_y

        sx = cw / 300.0
        sy = ch / 161.0

        def tc(px, py):
            return int(pad_x + px * sx), int(pad_y + py * sy)

        court_rect_tl = tc(0, 0)
        court_rect_br = tc(300, 161)

        # Fill neutral court background
        cv2.rectangle(canvas, court_rect_tl, court_rect_br, self.COURT_FILL, -1)

        # 1. Draw Voronoi Space Control Regions
        if voronoi_regions:
            for region in voronoi_regions:
                verts = region['vertices']
                team = region['team_id']
                scaled_verts = np.array([tc(v[0], v[1]) for v in verts], dtype=np.int32)
                fill_color = self.TEAM1_VORONOI if team == 1 else self.TEAM2_VORONOI
                cv2.fillPoly(canvas, [scaled_verts], fill_color)

        # 2. Draw Court Markings (Crisp, over Voronoi)
        # Perimeter outline
        cv2.rectangle(canvas, court_rect_tl, court_rect_br, self.COURT_LINE_COLOR, 1, cv2.LINE_AA)

        # Half-court line
        cv2.line(canvas, tc(150, 0), tc(150, 161), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)

        # Center circle
        center = tc(150, 80.5)
        center_rx = int(1.80 / self.court_m_w * 300 * sx)
        center_ry = int(1.80 / self.court_m_h * 161 * sy)
        cv2.ellipse(canvas, center, (center_rx, center_ry), 0, 0, 360,
                    self.COURT_LINE_COLOR, 1, cv2.LINE_AA)

        # Both half-court markings
        for side in ['left', 'right']:
            self._draw_half_court_markings(canvas, tc, sx, sy, side)

        # Reference point markers (small rings)
        ref_pts = [(50, 80.5), (250, 80.5)]
        for rx, ry in ref_pts:
            cv2.circle(canvas, tc(rx, ry), 5, self.REF_MARKER, 1, cv2.LINE_AA)

        # 3. Draw Player Dots
        for pid, pos in tactical_positions.items():
            team = player_assignment_frame.get(pid, 1)
            dot_color = self.TEAM1_DOT if team == 1 else self.TEAM2_DOT
            cx, cy = tc(pos[0], pos[1])

            # White outer ring
            cv2.circle(canvas, (cx, cy), 10, self.DOT_BORDER, 2, cv2.LINE_AA)
            # Team colored interior
            cv2.circle(canvas, (cx, cy), 8, dot_color, -1, cv2.LINE_AA)

            # Ball possession highlight
            if pid == ball_acquisition_frame:
                cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def _draw_half_court_markings(self, canvas, tc, sx, sy, side):
        """Draw key, free throw circle, restricted area, and 3-point lines."""
        # Key dimensions: 5.79m from baseline, width 4.88m (centered at y=80.5)
        key_depth = 5.79 / self.court_m_w * 300
        key_y1 = (80.5 - (4.88 / 2) / self.court_m_h * 161)
        key_y2 = (80.5 + (4.88 / 2) / self.court_m_h * 161)

        if side == 'left':
            kx1, kx2 = 0, key_depth
            ft_x = key_depth
        else:
            kx1, kx2 = 300 - key_depth, 300
            ft_x = 300 - key_depth

        # Paint rectangle
        cv2.rectangle(canvas, tc(kx1, key_y1), tc(kx2, key_y2), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)

        # Free throw circle
        ft_center = tc(ft_x, 80.5)
        ft_rx = int(1.80 / self.court_m_w * 300 * sx)
        ft_ry = int(1.80 / self.court_m_h * 161 * sy)
        cv2.ellipse(canvas, ft_center, (ft_rx, ft_ry), 0, 0, 360, self.COURT_LINE_COLOR, 1, cv2.LINE_AA)

        # Three-point arc
        three_corner_depth = 4.26 / self.court_m_w * 300
        three_y_top = 0.91 / self.court_m_h * 161
        three_y_bot = 161 - three_y_top

        basket_x = 1.575 / self.court_m_w * 300 if side == 'left' else (300 - 1.575 / self.court_m_w * 300)
        basket_center = tc(basket_x, 80.5)
        arc_rx = int(7.24 / self.court_m_w * 300 * sx)
        arc_ry = int(7.24 / self.court_m_h * 161 * sy)

        if side == 'left':
            # Straight corner lines
            cv2.line(canvas, tc(0, three_y_top), tc(three_corner_depth, three_y_top), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.line(canvas, tc(0, three_y_bot), tc(three_corner_depth, three_y_bot), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
            # Arc
            cv2.ellipse(canvas, basket_center, (arc_rx, arc_ry), 0, -68, 68, self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.line(canvas, tc(300, three_y_top), tc(300 - three_corner_depth, three_y_top), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.line(canvas, tc(300, three_y_bot), tc(300 - three_corner_depth, three_y_bot), self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.ellipse(canvas, basket_center, (arc_rx, arc_ry), 0, 112, 248, self.COURT_LINE_COLOR, 1, cv2.LINE_AA)
