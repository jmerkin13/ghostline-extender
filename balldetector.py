# Pink Ball & White Line Detection Script
# Production version - loads settings from JSON and displays detection overlay

# Import required libraries
import cv2  # OpenCV for image processing and display
import numpy as np  # NumPy for numerical operations on arrays
import mss  # MSS for fast screen capture
import time  # Time for FPS control
import json  # JSON for loading settings
import os    # OS for file path checking

# Define the monitor region to capture
# This specifies the exact screen area to grab frames from
monitor = {"left": 2151, "top": 75, "width": 1380, "height": 750}

# Target FPS for capture (45 frames per second)
TARGET_FPS = 45

# Calculate the delay between frames to achieve target FPS
# 1 second / 45 FPS = ~0.0222 seconds per frame
FRAME_DELAY = 1.0 / TARGET_FPS

# Configuration file for loading mask settings
CONFIG_FILE = "line_detection_settings.json"

# ROI constraint - search radius around ghostball for white line
ROI_RADIUS = 70

# Line filtering - max distance from ghostball center to line endpoint
MAX_ENDPOINT_DISTANCE = 5

# Rail coordinates for overlay
# Format: ((start_x, start_y), (end_x, end_y))
RAILS = [
    ((125, 80), (646, 80)),      # Top Left
    ((732, 80), (1252, 80)),     # Top Right
    ((125, 679), (646, 679)),    # Bottom Left
    ((732, 679), (1252, 679)),   # Bottom Right
    ((83, 125), (83, 631)),      # Left
    ((1295, 125), (1295, 631))   # Right
]

def get_circular_roi(image, center, radius):
    # Extract circular ROI around center point with boundary validation
    h, w = image.shape[:2]
    x, y = center

    # Calculate bounding box with boundary checks
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)

    # Validate ROI dimensions
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, None

    # Extract bounding box
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    return roi, (x1, y1, x2, y2)

def detect_white_line(roi_hsv, roi_rect, ghostball_center, settings):
    # Detect white line in ROI using HoughLinesP with loaded settings
    # Create white mask using settings
    white_lower = np.array(settings["hsv_lower"])
    white_upper = np.array(settings["hsv_upper"])
    white_mask = cv2.inRange(roi_hsv, white_lower, white_upper)

    # Detect lines with settings parameters
    lines = cv2.HoughLinesP(white_mask, 1, np.pi/180,
                           settings["line_threshold"],
                           minLineLength=settings["min_line_length"],
                           maxLineGap=settings["max_line_gap"])

    if lines is None:
        return None

    # Filter lines by proximity to ghostball
    x1_offset, y1_offset = roi_rect[0], roi_rect[1]
    min_dist = float('inf')
    best_line = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Convert to frame coordinates
        x1_frame = x1 + x1_offset
        y1_frame = y1 + y1_offset
        x2_frame = x2 + x1_offset
        y2_frame = y2 + y1_offset

        # Calculate distance from endpoints to ghostball
        dist1 = np.sqrt((x1_frame - ghostball_center[0])**2 +
                       (y1_frame - ghostball_center[1])**2)
        dist2 = np.sqrt((x2_frame - ghostball_center[0])**2 +
                       (y2_frame - ghostball_center[1])**2)

        min_endpoint_dist = min(dist1, dist2)

        # Select line closest to ghostball
        if min_endpoint_dist <= MAX_ENDPOINT_DISTANCE and min_endpoint_dist < min_dist:
            min_dist = min_endpoint_dist
            best_line = (x1_frame, y1_frame, x2_frame, y2_frame)

    return best_line

# Extend the detected white line to the frame edge
# The white line IS the aim line - just extend it in its existing direction
# Parameters:
#   - line: detected line segment (x1, y1, x2, y2) from HoughLinesP
#   - frame_width, frame_height: dimensions of the capture frame
#   - ghostball_center: (x, y) center position of the pink ball
# Returns: (x1, y1, x2, y2) where line extends from ghostball to frame edge
def extend_line_to_edges(line, frame_width, frame_height, ghostball_center):
    # Unpack detected line endpoints
    x1, y1, x2, y2 = line

    # Calculate direction vector from detected line segment
    # The white line already points in the correct aim direction
    dx = x2 - x1
    dy = y2 - y1

    # Avoid division by zero for degenerate lines (no length)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return line

    # Normalize direction vector to unit length
    length = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / length
    dy_norm = dy / length

    # Determine which endpoint is closer to ghostball
    gb_x, gb_y = ghostball_center
    dist1 = np.sqrt((x1 - gb_x)**2 + (y1 - gb_y)**2)
    dist2 = np.sqrt((x2 - gb_x)**2 + (y2 - gb_y)**2)

    # If endpoint 2 is closer, reverse direction (point away from closer endpoint)
    if dist2 < dist1:
        dx_norm = -dx_norm
        dy_norm = -dy_norm

    # Extend from ghostball in ONE direction only
    # Line equation: P(t) = ghostball + t * direction (t > 0 only)
    t_values = []

    # Check intersection with left and right frame edges
    if abs(dx_norm) > 1e-6:
        t_left = -gb_x / dx_norm  # Left edge (x=0)
        t_right = (frame_width - gb_x) / dx_norm  # Right edge
        if t_left > 0:
            t_values.append(t_left)
        if t_right > 0:
            t_values.append(t_right)

    # Check intersection with top and bottom frame edges
    if abs(dy_norm) > 1e-6:
        t_top = -gb_y / dy_norm  # Top edge (y=0)
        t_bottom = (frame_height - gb_y) / dy_norm  # Bottom edge
        if t_top > 0:
            t_values.append(t_top)
        if t_bottom > 0:
            t_values.append(t_bottom)

    # Find smallest positive t (first frame edge intersection)
    if t_values:
        t_min = min(t_values)
        end_x = int(gb_x + t_min * dx_norm)
        end_y = int(gb_y + t_min * dy_norm)
        # Return line FROM ghostball TO frame edge
        return (gb_x, gb_y, end_x, end_y)

    # Fallback: return line from ghostball to far endpoint
    return (gb_x, gb_y, x2, y2)

def load_settings():
    # Load settings from JSON file
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    # Default settings if file doesn't exist
    print(f"Warning: {CONFIG_FILE} not found, using default settings")
    return {
        "ghostball": {
            "hsv_lower": [140, 50, 50],
            "hsv_upper": [170, 255, 255],
            "min_area": 200,
            "max_area": 1000,
            "min_circularity": 70
        },
        "white_line": {
            "hsv_lower": [0, 0, 200],
            "hsv_upper": [180, 30, 255],
            "line_threshold": 30,
            "min_line_length": 20,
            "max_line_gap": 10
        }
    }

def main():
    # Print startup message
    print("=" * 60)
    print("BALLDETECTOR - PINK BALL & WHITE LINE DETECTION")
    print("=" * 60)
    print("Loading settings from", CONFIG_FILE)
    print("Press 'q' to quit")
    print("Press 'r' to toggle rails overlay")
    print("=" * 60)
    print()

    # Load detection settings
    settings = load_settings()
    print("Settings loaded successfully")
    print()

    with mss.mss() as sct:
        # Create display window
        window_name = "Balldetector - Live Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Toggle state for rails overlay
        show_rails = True

        while True:
            start_time = time.time()

            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # Create display frame
            display_frame = frame_bgr.copy()

            # Draw rails overlay if enabled
            if show_rails:
                for p1, p2 in RAILS:
                    cv2.line(display_frame, p1, p2, (0, 0, 0), 1)

            # Detect ghostball using contours
            gb_settings = settings["ghostball"]
            ghostball_lower = np.array(gb_settings["hsv_lower"])
            ghostball_upper = np.array(gb_settings["hsv_upper"])
            ghostball_mask = cv2.inRange(frame_hsv, ghostball_lower, ghostball_upper)

            # Find contours for ghostball
            contours, _ = cv2.findContours(ghostball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ghostball_center = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if gb_settings["min_area"] <= area <= gb_settings["max_area"]:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity >= (gb_settings["min_circularity"] / 100.0):
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            center = (int(x), int(y))
                            # Draw black circle overlay on detected ghostball
                            cv2.circle(display_frame, center, int(radius), (0, 0, 0), 1)
                            cv2.circle(display_frame, center, 1, (0, 0, 0), 1)
                            ghostball_center = center
                            break

            # Detect white line if ghostball found
            if ghostball_center is not None:
                # Extract ROI from HSV frame
                roi_hsv, roi_rect = get_circular_roi(frame_hsv, ghostball_center, ROI_RADIUS)
                if roi_hsv is not None and roi_rect is not None:
                    # Detect white line
                    line = detect_white_line(roi_hsv, roi_rect, ghostball_center, settings["white_line"])
                    if line is not None:
                        # Extend line from ghostball center in detected direction
                        extended_line = extend_line_to_edges(line, monitor["width"], monitor["height"], ghostball_center)
                        x1, y1, x2, y2 = extended_line
                        # Draw 1-pixel cyan line overlay
                        # BGR format: (255, 255, 0) = Cyan
                        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # Display frame
            cv2.imshow(window_name, display_frame)

            # Keyboard controls
            elapsed_time = time.time() - start_time
            remaining_delay = FRAME_DELAY - elapsed_time
            wait_time = max(1, int(remaining_delay * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                print("\nQuitting application...")
                break
            elif key == ord('r'):
                show_rails = not show_rails
                print(f"Rails overlay: {'On' if show_rails else 'Off'}")

        cv2.destroyAllWindows()
        print("Script terminated cleanly. All resources released.")

# Entry point of the script
# This ensures main() only runs when script is executed directly
# Not when imported as a module
if __name__ == "__main__":
    main()
