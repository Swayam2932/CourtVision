"""
Voronoi-based space control analysis for basketball.

Computes Voronoi tessellation of the court based on player positions,
then calculates what percentage of the frontcourt each team controls.
"""

import numpy as np
from scipy.spatial import Voronoi


class VoronoiSpaceControl:
    """
    Calculates Voronoi space control regions for each team on the court.
    
    Uses Voronoi tessellation to partition the court into regions closest to
    each player, then sums areas per team to produce control percentages.
    """

    def __init__(self, court_width_px=300, court_height_px=161,
                 court_width_m=28.0, court_height_m=15.0):
        self.court_width = court_width_px
        self.court_height = court_height_px
        self.court_width_m = court_width_m
        self.court_height_m = court_height_m

    def compute_voronoi_regions(self, tactical_positions, player_assignment_frame):
        """
        Compute bounded Voronoi regions on the court for all players.

        Args:
            tactical_positions (dict): {player_id: (x, y)} in tactical pixel coords.
            player_assignment_frame (dict): {player_id: team_id}.

        Returns:
            list: List of dicts with keys 'player_id', 'team_id', 'vertices' (Nx2 np array),
                  'area' (float in m²). Empty list if fewer than 3 players.
        """
        if len(tactical_positions) < 3:
            return []

        ids = list(tactical_positions.keys())
        points = np.array([tactical_positions[pid] for pid in ids], dtype=np.float64)

        # Mirror points across court boundaries to create bounded Voronoi
        mirror_points = []
        mirror_ids = []
        for i, (px, py) in enumerate(points):
            mirrors = [
                (-px, py), (2 * self.court_width - px, py),
                (px, -py), (px, 2 * self.court_height - py),
            ]
            for mp in mirrors:
                mirror_points.append(mp)
                mirror_ids.append(ids[i])

        all_points = np.vstack([points, np.array(mirror_points)])

        try:
            vor = Voronoi(all_points)
        except Exception:
            return []

        regions = []
        court_poly = np.array([
            [0, 0], [self.court_width, 0],
            [self.court_width, self.court_height], [0, self.court_height]
        ], dtype=np.float64)

        for i, pid in enumerate(ids):
            region_idx = vor.point_region[i]
            region_vertices_idx = vor.regions[region_idx]

            if -1 in region_vertices_idx or len(region_vertices_idx) == 0:
                continue

            verts = vor.vertices[region_vertices_idx]
            # Clip to court bounds
            clipped = self._clip_polygon_to_rect(verts, 0, self.court_width, 0, self.court_height)
            if clipped is None or len(clipped) < 3:
                continue

            area_px = self._polygon_area(clipped)
            # Convert pixel area to m²
            scale_x = self.court_width_m / self.court_width
            scale_y = self.court_height_m / self.court_height
            area_m2 = area_px * scale_x * scale_y

            team_id = player_assignment_frame.get(pid, 1)
            regions.append({
                'player_id': pid,
                'team_id': team_id,
                'vertices': clipped,
                'area': area_m2,
            })

        return regions

    def compute_frontcourt_control(self, regions, attacking_team, attacking_direction):
        """
        Calculate what % of the frontcourt each team controls.

        Args:
            regions (list): Output of compute_voronoi_regions.
            attacking_team (int): 1 or 2.
            attacking_direction (str): 'left' or 'right' — which basket the
                attacking team is going towards.

        Returns:
            dict: {'team1_pct': float, 'team2_pct': float} both 0-100.
        """
        if not regions:
            return {'team1_pct': 50.0, 'team2_pct': 50.0}

        # Frontcourt is the half the attacking team is attacking into
        if attacking_direction == 'right':
            fc_x_min, fc_x_max = self.court_width / 2, self.court_width
        else:
            fc_x_min, fc_x_max = 0, self.court_width / 2

        team1_area = 0.0
        team2_area = 0.0

        for r in regions:
            clipped = self._clip_polygon_to_rect(
                r['vertices'], fc_x_min, fc_x_max, 0, self.court_height
            )
            if clipped is None or len(clipped) < 3:
                continue
            area = self._polygon_area(clipped)
            scale_x = self.court_width_m / self.court_width
            scale_y = self.court_height_m / self.court_height
            area_m2 = area * scale_x * scale_y

            if r['team_id'] == 1:
                team1_area += area_m2
            else:
                team2_area += area_m2

        total = team1_area + team2_area
        if total < 0.01:
            return {'team1_pct': 50.0, 'team2_pct': 50.0}

        return {
            'team1_pct': round(team1_area / total * 100, 1),
            'team2_pct': round(team2_area / total * 100, 1),
        }

    @staticmethod
    def _polygon_area(vertices):
        """Shoelace formula for polygon area."""
        n = len(vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _clip_polygon_to_rect(polygon, x_min, x_max, y_min, y_max):
        """Sutherland-Hodgman algorithm to clip polygon to rectangle."""
        if polygon is None or len(polygon) < 3:
            return None

        def clip_edge(poly, edge_start, edge_end):
            """Clip polygon against one edge using Sutherland-Hodgman."""
            if len(poly) == 0:
                return []
            clipped = []
            ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]

            for i in range(len(poly)):
                curr = poly[i]
                prev = poly[i - 1]

                curr_side = ex * (curr[1] - edge_start[1]) - ey * (curr[0] - edge_start[0])
                prev_side = ex * (prev[1] - edge_start[1]) - ey * (prev[0] - edge_start[0])

                if curr_side >= 0:
                    if prev_side < 0:
                        clipped.append(_intersect(prev, curr, edge_start, edge_end))
                    clipped.append(curr)
                elif prev_side >= 0:
                    clipped.append(_intersect(prev, curr, edge_start, edge_end))

            return clipped

        def _intersect(p1, p2, p3, p4):
            """Line segment intersection."""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return p2
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        poly = [tuple(p) for p in polygon]
        edges = [
            ((x_min, y_min), (x_max, y_min)),  # top
            ((x_max, y_min), (x_max, y_max)),  # right
            ((x_max, y_max), (x_min, y_max)),  # bottom
            ((x_min, y_max), (x_min, y_min)),  # left
        ]

        for edge_start, edge_end in edges:
            poly = clip_edge(poly, edge_start, edge_end)
            if len(poly) == 0:
                return None

        return np.array(poly, dtype=np.float64) if poly else None
