"""
CourtVision Pro — Basketball AI Analytics Pipeline

Produces a professional 1080×1350 portrait broadcast analytics video:
- Skeleton wireframe player overlays with speed badges
- Full-court dark-themed tactical view with Voronoi space control
- 4-card stats dashboard (Space Control, Team Spacing, Distance Covered, Open Space)
- Live attacking team & direction pill
- Possession recap summary at clip end
"""

import os
import sys

# Ensure config directories are writable across all environments
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_config")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

import argparse
import cv2
import numpy as np

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator

# Advanced Analytics Modules
from analytics import (
    VoronoiSpaceControl,
    TeamSpacingCalculator,
    OpenSpaceCalculator,
    PossessionTracker,
)

# Professional Visual Drawers
from drawers import (
    ProOverlayDrawer,
    ProTacticalDrawer,
    StatsDashboardDrawer,
    PossessionRecapDrawer,
    ProCanvasComposer,
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
)

from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    PRO_OUTPUT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    PRO_FPS,
    TEAM_1_NAME,
    TEAM_2_NAME,
    TEAM_1_COLOR_BGR,
    TEAM_2_COLOR_BGR,
    RECAP_DURATION_SECONDS,
)


def parse_args():
    parser = argparse.ArgumentParser(description='CourtVision Pro — Basketball AI Analytics')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    parser.add_argument('--team1', type=str, default=TEAM_1_NAME,
                        help='Team 1 display name (default: "PHI 76ERS")')
    parser.add_argument('--team2', type=str, default=TEAM_2_NAME,
                        help='Team 2 display name (default: "IND PACERS")')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy 720p output mode (original drawers)')
    return parser.parse_args()


def run_legacy_pipeline(video_frames, player_tracks, ball_tracks, court_keypoints_per_frame,
                        player_assignment, ball_acquisition, tactical_player_positions,
                        player_distances_per_frame, player_speed_per_frame,
                        tactical_view_converter, output_path):
    """Fallback legacy drawing pipeline."""
    pass_detector = PassAndInterceptionDetector()
    passes = pass_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_detector.detect_interceptions(ball_acquisition, player_assignment)

    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    keypoint_drawer = CourtKeypointDrawer()
    team_control_drawer = TeamBallControlDrawer()
    frame_num_drawer = FrameNumberDrawer()
    pass_drawer = PassInterceptionDrawer()
    tac_drawer = TacticalViewDrawer()
    speed_drawer = SpeedAndDistanceDrawer()

    frames = player_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)
    frames = ball_drawer.draw(frames, ball_tracks)
    frames = keypoint_drawer.draw(frames, court_keypoints_per_frame)
    frames = frame_num_drawer.draw(frames)
    frames = team_control_drawer.draw(frames, player_assignment, ball_acquisition)
    frames = pass_drawer.draw(frames, passes, interceptions)
    frames = speed_drawer.draw(frames, player_tracks, player_distances_per_frame, player_speed_per_frame)
    frames = tac_drawer.draw(
        frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_acquisition,
    )
    save_video(frames, output_path)
    print(f"✓ Legacy output saved to: {output_path}")


def main():
    args = parse_args()

    team_1_name = args.team1
    team_2_name = args.team2

    # Determine default output path
    if args.output_video is None:
        args.output_video = OUTPUT_VIDEO_PATH if args.legacy else PRO_OUTPUT_VIDEO_PATH

    print(f"[1/8] Reading video: {args.input_video}")
    video_frames = read_video(args.input_video)
    num_frames = len(video_frames)
    print(f"       Loaded {num_frames} frames")

    # Read original FPS
    cap = cv2.VideoCapture(args.input_video)
    input_fps = cap.get(cv2.CAP_PROP_FPS) or PRO_FPS
    cap.release()

    # ─── 1. Player & Ball Detection / Tracking ────────────────────────
    print("[2/8] Running player & ball detection with tracking...")
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    player_tracks = player_tracker.get_object_tracks(
        video_frames, read_from_stub=True,
        stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl'))

    ball_tracks = ball_tracker.get_object_tracks(
        video_frames, read_from_stub=True,
        stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl'))

    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames, read_from_stub=True,
        stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl'))

    # Post-process ball
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # ─── 2. Team Assignment ───────────────────────────────────────────
    print("[3/8] Assigning player teams...")
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames, player_tracks, read_from_stub=True,
        stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl'))

    # ─── 3. Ball Possession Detection ─────────────────────────────────
    print("[4/8] Detecting ball possession...")
    ball_acquisition_detector = BallAquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # ─── 4. Dynamic Camera Calibration & Tactical Mapping ─────────────
    print("[5/8] Computing court homography & tactical positions...")
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png")
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, player_tracks)

    # ─── 5. Speed & Distance Calculations ─────────────────────────────
    print("[6/8] Calculating real-world speed & distance in metres...")
    speed_distance_calc = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters)
    player_distances_per_frame = speed_distance_calc.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_distance_calc.calculate_speed(player_distances_per_frame, fps=input_fps)

    # Check for legacy mode
    if args.legacy:
        print("[7/8] Running legacy drawers...")
        run_legacy_pipeline(video_frames, player_tracks, ball_tracks, court_keypoints_per_frame,
                            player_assignment, ball_acquisition, tactical_player_positions,
                            player_distances_per_frame, player_speed_per_frame,
                            tactical_view_converter, args.output_video)
        return

    # ─── 6. Advanced Analytics (Voronoi, Spacing, Open Space, Recap) ──
    print("[7/8] Running advanced spatial analytics...")

    # Attacking team and direction
    possession_tracker = PossessionTracker(fps=input_fps)
    attacking_teams, attacking_directions = possession_tracker.track_possessions_across_frames(
        tactical_player_positions, player_assignment, ball_acquisition)

    # Voronoi Space Control
    voronoi_calc = VoronoiSpaceControl()
    voronoi_regions_per_frame = []
    frontcourt_pcts_per_frame = []
    for frame_idx in range(num_frames):
        tac = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else {}
        assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
        atk = attacking_teams[frame_idx]
        atk_dir = attacking_directions[frame_idx]

        regions = voronoi_calc.compute_voronoi_regions(tac, assign)
        voronoi_regions_per_frame.append(regions)

        fc_pct = voronoi_calc.compute_frontcourt_control(regions, atk, atk_dir)
        frontcourt_pcts_per_frame.append(fc_pct)

    # Team Spacing (Convex Hull m²)
    spacing_calc = TeamSpacingCalculator()
    team_spacing_per_frame = spacing_calc.calculate_spacing_per_frame(
        tactical_player_positions, player_assignment)

    # Open Space (Distance to nearest defender in m)
    open_space_calc = OpenSpaceCalculator()
    open_space_per_frame = open_space_calc.calculate_open_space_per_frame(
        tactical_player_positions, player_assignment, attacking_teams)

    # Cumulative distances
    cum_dist_t1 = []
    cum_dist_t2 = []
    total_t1 = 0.0
    total_t2 = 0.0
    for frame_idx, dists in enumerate(player_distances_per_frame):
        assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
        for pid, d in dists.items():
            team = assign.get(pid, 1)
            if team == 1:
                total_t1 += d
            else:
                total_t2 += d
        cum_dist_t1.append(total_t1)
        cum_dist_t2.append(total_t2)

    # Possession Recap summary
    recap = possession_tracker.generate_possession_recap(
        tactical_player_positions, player_assignment, ball_acquisition,
        player_distances_per_frame, player_speed_per_frame,
        attacking_teams, attacking_directions,
        frontcourt_pcts_per_frame, open_space_per_frame)

    # ─── 7. Professional Rendering & Composition ──────────────────────
    print("[8/8] Rendering professional 1080×1350 broadcast analytics layout...")

    overlay_drawer = ProOverlayDrawer(
        team_1_color=list(TEAM_1_COLOR_BGR),
        team_2_color=list(TEAM_2_COLOR_BGR),
        team_1_name=team_1_name,
        team_2_name=team_2_name)

    tactical_drawer = ProTacticalDrawer()
    dashboard_drawer = StatsDashboardDrawer()
    recap_drawer = PossessionRecapDrawer(team_1_name=team_1_name, team_2_name=team_2_name)
    composer = ProCanvasComposer(team_1_name=team_1_name, team_2_name=team_2_name)

    # Draw skeleton overlays and badges on video frames
    broadcast_frames = overlay_drawer.draw(
        video_frames, player_tracks, player_assignment,
        ball_acquisition, player_speed_per_frame)

    output_frames = []
    # Recap shown only on the last portion of video (at most 20% of clip duration or RECAP_DURATION_SECONDS)
    max_recap_frames = max(1, int(num_frames * 0.20))
    recap_frame_count = min(int(RECAP_DURATION_SECONDS * input_fps), max_recap_frames)

    for frame_idx in range(num_frames):
        # End of clip recap screen
        is_recap = frame_idx >= (num_frames - recap_frame_count)
        if is_recap:
            broadcast_layer = recap_drawer.draw(broadcast_frames[frame_idx], recap)
        else:
            broadcast_layer = broadcast_frames[frame_idx]

        # Tactical view layer
        tac = tactical_player_positions[frame_idx] if frame_idx < len(tactical_player_positions) else {}
        assign = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
        ba = ball_acquisition[frame_idx] if frame_idx < len(ball_acquisition) else -1
        regions = voronoi_regions_per_frame[frame_idx]
        atk = attacking_teams[frame_idx]
        atk_dir = attacking_directions[frame_idx]

        tactical_layer = tactical_drawer.draw_tactical_frame(
            tac, assign, ba, regions, atk, atk_dir,
            team_1_name=team_1_name, team_2_name=team_2_name)

        # Dashboard layer
        fc_pct = frontcourt_pcts_per_frame[frame_idx]
        spacing = team_spacing_per_frame[frame_idx]
        d1 = cum_dist_t1[frame_idx]
        d2 = cum_dist_t2[frame_idx]
        os_m = open_space_per_frame[frame_idx]

        dashboard_layer = dashboard_drawer.draw(
            fc_pct, spacing, d1, d2, os_m, atk,
            team_1_name=team_1_name, team_2_name=team_2_name)

        # Compose into single 1080x1350 canvas
        final_canvas = composer.compose(
            broadcast_layer, tactical_layer, dashboard_layer,
            attacking_team=atk, attacking_direction=atk_dir)
        output_frames.append(final_canvas)

        if frame_idx % 50 == 0:
            print(f"       Rendered {frame_idx}/{num_frames} frames")

    print(f"       Rendered {num_frames}/{num_frames} frames")

    # ─── Save Video Output ────────────────────────────────────────────
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ext = os.path.splitext(args.output_video)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext == '.mp4' else cv2.VideoWriter_fourcc(*'XVID')

    out_h, out_w = output_frames[0].shape[:2]
    writer = cv2.VideoWriter(args.output_video, fourcc, input_fps, (out_w, out_h))
    for frame in output_frames:
        writer.write(frame)
    writer.release()

    print(f"\n============================================================")
    print(f"✓ Video successfully generated: {args.output_video}")
    print(f"  Canvas: {out_w}×{out_h} (Portrait)")
    print(f"  Frame rate: {input_fps} fps ({num_frames} frames total)")
    print(f"  Teams: {team_1_name} vs {team_2_name}")
    print(f"  Possessions tracked: {recap['num_possessions']}")
    print(f"  Top Speed: {recap['top_speed_kmh']} km/h")
    print(f"============================================================")


if __name__ == '__main__':
    main()
