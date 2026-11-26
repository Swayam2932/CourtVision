"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles, ellipses, and rounded rectangles on frames,
used to represent annotations such as player wireframes, speed badges, and tactical UI cards.
"""

import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, thickness=-1):
    """
    Draw a filled or stroked rounded rectangle on an image.
    """
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    w = x2 - x1
    h = y2 - y1
    r = max(1, min(radius, w // 2, h // 2))

    if thickness == -1:
        # Filled rounded rectangle
        cv2.circle(image, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        # Outlined rounded rectangle
        cv2.line(image, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    return image


def draw_traingle(frame, bbox, color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.
    """
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draws an ellipse and an optional rectangle with a track ID on the given frame.
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame
