STUBS_DEFAULT_PATH = 'stubs'
PLAYER_DETECTOR_PATH = 'models/player_detector.pt'
BALL_DETECTOR_PATH = 'models/ball_detector_model.pt'
COURT_KEYPOINT_DETECTOR_PATH = 'models/court_keypoint_detector.pt'
OUTPUT_VIDEO_PATH = 'output_videos/output_video.avi'

# Professional portrait layout settings (1080x1350)
PRO_OUTPUT_VIDEO_PATH = 'output_videos/pro_output_video.mp4'
PRO_CANVAS_WIDTH = 1080
PRO_CANVAS_HEIGHT = 1350
PRO_FPS = 30

# Default team display names (matches reference video: Philadelphia 76ers vs Indiana Pacers)
TEAM_1_NAME = "PHI 76ERS"
TEAM_2_NAME = "IND PACERS"

# Team colors in BGR format
TEAM_1_COLOR_BGR = (240, 160, 30)    # Cyan-blue (PHI)
TEAM_2_COLOR_BGR = (20, 190, 255)    # Amber-gold (IND)

# Duration (in seconds) to display the possession recap overlay at the end of the video
RECAP_DURATION_SECONDS = 4
