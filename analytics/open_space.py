"""
Open space calculator for basketball analytics.

Calculates the average gap (in metres) between each offensive player
and their nearest defender — a measure of how 'open' the offense is.
"""

import numpy as np


class OpenSpaceCalculator:
    """
    Computes average nearest-defender distance for the attacking team.
    """

    def __init__(self, court_width_px=300, court_height_px=161,
                 court_width_m=28.0, court_height_m=15.0):
        self.scale_x = court_width_m / court_width_px
        self.scale_y = court_height_m / court_height_px

    def calculate_open_space(self, tactical_positions, player_assignment_frame, attacking_team):
        """
        Calculate average distance from each offensive player to nearest defender.

        Args:
            tactical_positions (dict): {player_id: (x, y)} in pixel coords.
            player_assignment_frame (dict): {player_id: team_id}.
            attacking_team (int): 1 or 2.

        Returns:
            float: Average gap in metres. 0.0 if insufficient data.
        """
        offense_pts = []
        defense_pts = []

        for pid, pos in tactical_positions.items():
            team = player_assignment_frame.get(pid, 1)
            pos_m = np.array([pos[0] * self.scale_x, pos[1] * self.scale_y])
            if team == attacking_team:
                offense_pts.append(pos_m)
            else:
                defense_pts.append(pos_m)

        if not offense_pts or not defense_pts:
            return 0.0

        offense_arr = np.array(offense_pts)
        defense_arr = np.array(defense_pts)

        nearest_dists = []
        for op in offense_arr:
            dists = np.linalg.norm(defense_arr - op, axis=1)
            nearest_dists.append(np.min(dists))

        return round(float(np.mean(nearest_dists)), 1)

    def calculate_open_space_per_frame(self, tactical_player_positions,
                                       player_assignment, attacking_teams):
        """
        Calculate open space metric for each frame.

        Args:
            tactical_player_positions (list[dict]): Per-frame positions.
            player_assignment (list[dict]): Per-frame assignments.
            attacking_teams (list[int]): Per-frame attacking team (1 or 2).

        Returns:
            list[float]: Per-frame open space in metres.
        """
        results = []
        for frame_idx in range(len(tactical_player_positions)):
            tac = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else {}
            assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            atk = attacking_teams[frame_idx] if frame_idx < len(attacking_teams) else 1
            results.append(self.calculate_open_space(tac, assign, atk))
        return results
