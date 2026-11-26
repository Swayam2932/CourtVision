"""
Possession tracking and attacking direction detection.

Determines which team is attacking, which direction they're going,
and tracks possession changes for recap generation.
"""

import numpy as np


class PossessionTracker:
    """
    Tracks possessions, attacking direction, and generates possession recaps.
    """

    def __init__(self, court_width_px=300, court_height_px=161,
                 court_width_m=28.0, court_height_m=15.0, fps=30):
        self.court_width = court_width_px
        self.court_height = court_height_px
        self.court_width_m = court_width_m
        self.court_height_m = court_height_m
        self.fps = fps

    def detect_attacking_team_and_direction(self, tactical_positions, player_assignment_frame,
                                             ball_acquisition_frame):
        """
        Determine which team is attacking and their direction.

        Uses ball possession and player centroid positions:
        - The team with the ball is attacking
        - Direction is towards the basket their centroid is closer to
        
        Args:
            tactical_positions (dict): {player_id: (x, y)}.
            player_assignment_frame (dict): {player_id: team_id}.
            ball_acquisition_frame (int): player_id with ball, or -1.

        Returns:
            tuple: (attacking_team: int, direction: str)
                   direction is 'left' or 'right'
        """
        if ball_acquisition_frame == -1:
            return 1, 'left'

        attacking_team = player_assignment_frame.get(ball_acquisition_frame, 1)

        # Determine direction by centroid of attacking team
        atk_positions = []
        for pid, pos in tactical_positions.items():
            team = player_assignment_frame.get(pid, 1)
            if team == attacking_team:
                atk_positions.append(pos)

        if not atk_positions:
            return attacking_team, 'left'

        centroid_x = np.mean([p[0] for p in atk_positions])
        # If centroid is on the left half, they're attacking right (towards right basket)
        # If centroid is on the right half, they're attacking left
        direction = 'right' if centroid_x < self.court_width / 2 else 'left'

        return attacking_team, direction

    def track_possessions_across_frames(self, tactical_player_positions,
                                         player_assignment, ball_acquisition):
        """
        Determine attacking team and direction for every frame.

        Args:
            tactical_player_positions (list[dict]): Per-frame positions.
            player_assignment (list[dict]): Per-frame assignments.
            ball_acquisition (list[int]): Per-frame ball holder.

        Returns:
            tuple: (attacking_teams: list[int], directions: list[str])
        """
        attacking_teams = []
        directions = []

        last_team = 1
        last_dir = 'left'

        for frame_idx in range(len(ball_acquisition)):
            tac = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else {}
            assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            ba = ball_acquisition[frame_idx]

            if ba != -1 and tac:
                team, direction = self.detect_attacking_team_and_direction(tac, assign, ba)
                last_team = team
                last_dir = direction
            
            attacking_teams.append(last_team)
            directions.append(last_dir)

        return attacking_teams, directions

    def generate_possession_recap(self, tactical_player_positions, player_assignment,
                                   ball_acquisition, player_distances_per_frame,
                                   player_speed_per_frame, attacking_teams, directions,
                                   voronoi_frontcourt_pcts, open_space_per_frame):
        """
        Generate a summary recap of all possessions in the clip.

        Args:
            Various per-frame data structures.

        Returns:
            dict: Recap with keys:
                - 'num_possessions' (int)
                - 'duration_seconds' (float)
                - 'team1_distance_m' (float)
                - 'team2_distance_m' (float)
                - 'top_speed_kmh' (float)
                - 'top_speed_team' (int)
                - 'team1_frontcourt_pct' (float)
                - 'team2_frontcourt_pct' (float)
                - 'team1_avg_open_space' (float)
                - 'team2_avg_open_space' (float)
        """
        num_frames = len(ball_acquisition)
        duration = num_frames / self.fps

        # Count possessions (changes in attacking team)
        num_possessions = 1
        for i in range(1, len(attacking_teams)):
            if attacking_teams[i] != attacking_teams[i - 1]:
                num_possessions += 1

        # Total distance per team
        team1_total_dist = 0.0
        team2_total_dist = 0.0
        for frame_idx, dists in enumerate(player_distances_per_frame):
            assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            for pid, d in dists.items():
                team = assign.get(pid, 1)
                if team == 1:
                    team1_total_dist += d
                else:
                    team2_total_dist += d

        # Top speed (filtering tracking jump outliers)
        top_speed = 0.0
        top_speed_team = 1
        for frame_idx, speeds in enumerate(player_speed_per_frame):
            assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
            for pid, s in speeds.items():
                if 0 < s <= 32.0 and s > top_speed:
                    top_speed = s
                    top_speed_team = assign.get(pid, 1)

        # Average frontcourt control
        t1_fc = []
        t2_fc = []
        for fc in voronoi_frontcourt_pcts:
            t1_fc.append(fc.get('team1_pct', 50.0))
            t2_fc.append(fc.get('team2_pct', 50.0))

        # Average open space per attacking stint
        t1_open = []
        t2_open = []
        for frame_idx, os_val in enumerate(open_space_per_frame):
            atk = attacking_teams[frame_idx] if frame_idx < len(attacking_teams) else 1
            if os_val > 0:
                if atk == 1:
                    t1_open.append(os_val)
                else:
                    t2_open.append(os_val)

        return {
            'num_possessions': num_possessions,
            'duration_seconds': round(duration, 1),
            'team1_distance_m': round(team1_total_dist, 0),
            'team2_distance_m': round(team2_total_dist, 0),
            'top_speed_kmh': round(top_speed, 1),
            'top_speed_team': top_speed_team,
            'team1_frontcourt_pct': round(np.mean(t1_fc), 0) if t1_fc else 50,
            'team2_frontcourt_pct': round(np.mean(t2_fc), 0) if t2_fc else 50,
            'team1_avg_open_space': round(np.mean(t1_open), 1) if t1_open else 0.0,
            'team2_avg_open_space': round(np.mean(t2_open), 1) if t2_open else 0.0,
        }
