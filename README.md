# CourtVision

CourtVision is a Python computer-vision pipeline for basketball videos. It detects and tracks players and the ball, estimates which team has possession, identifies passes and interceptions, estimates player speed and distance, and adds a tactical top-down court view to an annotated output video.

The project is a command-line application. It processes one input video per run and writes one annotated video.

## What you need

- Python 3.9 (the Docker image uses Python 3.9)
- The Python packages in `requirements.txt`
- The three model weight files listed below
- A basketball video readable by OpenCV/FFmpeg
- Internet access on the first run if the Hugging Face `fashion-clip` model has not already been cached (used for jersey color/team assignment)

The included detector weights are large (about 728 MB total). They and the sample videos are not stored in the GitHub repository. Download the weights from the [CourtVision Google Drive folder](https://drive.google.com/drive/folders/1GfzD-WVNPAB4aDNuNdy90bXJgHVy3NOH?usp=sharing).

## 1. Get the project code

Clone the repository and enter its directory:

```bash
git clone https://github.com/Swayam2932/CourtVision.git
cd CourtVision
```

All commands below assume you are running them from the `CourtVision` project root (the directory containing `main.py`).

## 2. Download and place the model weights

Create a `models` directory in the project root if it does not already exist. Download the three `.pt` files from Drive and put them directly inside that directory. The names must match exactly:

```text
CourtVision/
├── main.py
└── models/
    ├── player_detector.pt
    ├── ball_detector_model.pt
    └── court_keypoint_detector.pt
```

The code loads these relative paths (configured in `configs/configs.py`):

| File | Used for |
| --- | --- |
| `models/player_detector.pt` | Player detection and tracking |
| `models/ball_detector_model.pt` | Ball detection and tracking |
| `models/court_keypoint_detector.pt` | Basketball court keypoint detection |

If Drive downloads a ZIP, extract it and move/rename the files so that the three paths above exist. Do not put the files inside an extra nested directory such as `models/models/`.

## 3. Add an input video

Create `input_videos/` in the project root and put a video there, for example:

```text
CourtVision/
├── input_videos/
│   └── game.mp4
├── models/
│   ├── player_detector.pt
│   ├── ball_detector_model.pt
│   └── court_keypoint_detector.pt
└── main.py
```

The input video can be stored elsewhere; just pass its path to the command. The `input_videos/` folder and videos are ignored by Git, so they will not appear after cloning unless you add them yourself.

## 4. Install dependencies

### macOS / Linux

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.9` is not installed, install Python 3.9 first. On Linux, you may also need FFmpeg and OpenCV system libraries. Docker is an alternative described below.

### Windows PowerShell

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Run the analysis

From the repository root, run:

```bash
python main.py input_videos/game.mp4
```

The default output path is:

```text
output_videos/output_video.avi
```

You can choose another output filename/path:

```bash
python main.py input_videos/game.mp4 --output_video output_videos/game_analysis.avi
```

The output directory is created automatically if needed. To see the command options:

```bash
python main.py --help
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `input_video` | Yes | — | Path to the source video |
| `--output_video` | No | `output_videos/output_video.avi` | Where to write the annotated video |
| `--stub_path` | No | `stubs` | Directory for intermediate cached analysis files |

Example using a video outside the project folder:

```bash
python main.py /path/to/my_game.mp4 --output_video output_videos/my_game_analysis.avi
```

## Run with Docker

Build the image from the project root:

```bash
docker build -t courtvision .
```

The Docker build excludes the large `models/` folder, so mount the downloaded weights, input video folder, and output folder when running the container:

```bash
mkdir -p models input_videos output_videos
# Put the video in input_videos/ and the three weight files in models/ first.

docker run --rm \
  -v "$PWD/models:/app/models" \
  -v "$PWD/input_videos:/app/input_videos" \
  -v "$PWD/output_videos:/app/output_videos" \
  courtvision input_videos/game.mp4 --output_video output_videos/game_analysis.avi
```

The paths after `courtvision` are paths inside the container. The mounted folders make the weights and videos available inside it and keep the output on your host machine.

## What the output contains

The pipeline overlays player track IDs and team colors, ball tracking, court keypoints, frame numbers, team ball control, pass/interception indicators, and player speed/distance information. It also draws a tactical court view showing mapped player positions.

## Intermediate cache files

The pipeline checks `stubs/` for cached player tracks, ball tracks, court keypoints, and team assignments. If a cache is missing or does not match the input video's frame count, that step runs again and writes a cache. The cache folder is generated locally and is not needed to start a fresh run. Do not reuse a cache from a different video unless you know it matches the video and frame count.

## Project layout

```text
ball_aquisition/                  Ball possession estimation
configs/                           Model, cache, and output paths
court_keypoint_detector/          Court keypoint detection
drawers/                            Video overlays and visualizations
images/                            Court image and court dimension references
input_videos/                      Your input videos (create this locally)
models/                            Three required detector weights (download separately)
output_videos/                     Generated annotated videos
pass_and_interception_detector/   Pass and interception detection
speed_and_distance_calculator/   Distance and speed estimation
tactical_view_converter/          Court coordinate mapping
team_assigner/                     Team assignment
trackers/                          Player and ball tracking
utils/                             Video and cache helpers
main.py                            Command-line pipeline
requirements.txt                   Python dependencies
Dockerfile                         Docker image definition
```

## Notes

- Run from the repository root because model and court image paths are relative to it.
- The complete input video is loaded into memory. Start with a short clip if memory is limited.
- CPU execution is supported but can take a long time; an available compatible GPU may speed up inference.
- The first uncached run downloads `patrickjohncyh/fashion-clip` through Hugging Face Transformers for team color assignment.
- `output_videos/`, `input_videos/`, and `stubs/` are ignored by Git. Keep your local input and result files backed up separately if needed.
