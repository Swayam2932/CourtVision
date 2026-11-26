"""
Team spacing calculator using convex hull area.

Measures how spread out each team is by computing the area of the
convex hull formed by that team's players on the tactical view.
"""

import numpy as np
from scipy.spatial import ConvexHull


class TeamSpacingCalculator:
    """
    Computes team spacing (convex hull area in m²) for each team per frame.
    """

    def __init__(self, court_width_px=300, court_height_px=161,
                 court_width_m=28.0, court_height_m=15.0):
        self.scale_x = court_width_m / court_width_px
        self.scale_y = court_height_m / court_height_px

    def calculate_spacing(self, tactical_positions, player_assignment_frame):
        """
        Calculate convex hull area for each team.

        Args:
            tactical_positions (dict): {player_id: (x, y)} in pixel coords.
            player_assignment_frame (dict): {player_id: team_id}.

        Returns:
            dict: {'team1_hull_m2': float, 'team2_hull_m2': float}
        """
        team1_points = []
        team2_points = []

        for pid, pos in tactical_positions.items():
            team = player_assignment_frame.get(pid, 1)
            # Convert to meters
            pos_m = (pos[0] * self.scale_x, pos[1] * self.scale_y)
            if team == 1:
                team1_points.append(pos_m)
            else:
                team2_points.append(pos_m)

        return {
            'team1_hull_m2': self._hull_area(team1_points),
            'team2_hull_m2': self._hull_area(team2_points),
        }

    @staticmethod
    def _hull_area(points):
        """Calculate convex hull area of a set of 2D points."""
        if len(points) < 3:
            return 0.0
        try:
            pts = np.array(points, dtype=np.float64)
            hull = ConvexHull(pts)
            return round(hull.volume, 1)  # In 2D, volume = area
        except Exception:
            return 0.0

    def calculate_spacing_per_frame(self, tactical_player_positions, player_assignment):
        """
        Calculate team spacing for all frames.

        Args:
            tactical_player_positions (list[dict]): Per-frame tactical positions.
            player_assignment (list[dict]): Per-frame team assignments.

        Returns:
            list[dict]: Per-frame spacing dicts.
        """
        spacings = []
        for frame_idx in range(len(tactical_player_positions)):
            tac = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else {}
            assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            spacings.append(self.calculate_spacing(tac, assign))
        return spacings
