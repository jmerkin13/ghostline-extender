# Pink Ball & White Line Detection Script
# =========================================
# This script is designed to overlay aim assistance for '8 Ball Pool' by detecting
# the game's ghostball (prediction marker) and the short aim line, then extending
# that line to show the full trajectory against the table rails.
#
# Production version - loads settings from JSON and displays detection overlay.

# Import required libraries
import cv2  # OpenCV (Open Source Computer Vision Library) used for all image processing, color conversion, and drawing.
import numpy as np  # NumPy is the fundamental package for scientific computing with Python, used for array manipulations.
import mss  # MSS is a fast cross-platform screen shot module, used for capturing the game window efficiently.
import time  # Standard time library, used for controlling the loop timing (FPS).
import json  # JSON library for parsing the configuration file.
import os    # OS library for checking file existence and paths.

# Define the monitor region to capture
# This dictionary specifies the exact screen coordinates to grab frames from.
# These values are specific to the user's screen resolution and window position.
# 'left', 'top': Top-left corner coordinates of the capture area.
# 'width', 'height': Dimensions of the capture area.
monitor = {"left": 2151, "top": 75, "width": 1380, "height": 750}

# Target Frames Per Second (FPS) for the main loop
# 45 FPS is chosen as a balance between responsiveness and CPU usage.
TARGET_FPS = 45

# Calculate the delay between frames to achieve target FPS
# Frame delay is the inverse of FPS. For 45 FPS: 1 / 45 ~= 0.022 seconds.
FRAME_DELAY = 1.0 / TARGET_FPS

# Configuration file name
# This file contains the HSV color ranges and other detection parameters.
CONFIG_FILE = "line_detection_settings.json"

# Region of Interest (ROI) constraint
# When searching for the white aim line, we restrict the search to a circle of this radius
# around the detected ghostball. This optimizes performance and reduces false positives.
ROI_RADIUS = 70

# Line filtering parameter
# When multiple lines are detected, we filter them based on how close their endpoint is
# to the ghostball center. Lines starting further than 5 pixels away are likely noise.
MAX_ENDPOINT_DISTANCE = 5

# Table Rail Coordinates
# These coordinates define the boundaries of the pool table within the captured frame.
# Used to calculate where the extended aim line should stop (simulating a bank or hit).
# Format: ((start_x, start_y), (end_x, end_y))
RAILS = [
    ((125, 80), (646, 80)),      # Top Left Rail
    ((732, 80), (1252, 80)),     # Top Right Rail
    ((125, 679), (646, 679)),    # Bottom Left Rail
    ((732, 679), (1252, 679)),   # Bottom Right Rail
    ((83, 125), (83, 631)),      # Left Rail
    ((1295, 125), (1295, 631))   # Right Rail
]

# Pre-computed table bounding box from RAILS (performance optimization)
# This is constant and never changes, so we calculate it once at module load
TABLE_MIN_X = min(min(start[0], end[0]) for start, end in RAILS)
TABLE_MAX_X = max(max(start[0], end[0]) for start, end in RAILS)
TABLE_MIN_Y = min(min(start[1], end[1]) for start, end in RAILS)
TABLE_MAX_Y = max(max(start[1], end[1]) for start, end in RAILS)

# Pre-computed rail bounding boxes for fast collision detection
# Each rail gets a pre-computed bounding box with tolerance applied
RAIL_BOUNDS = []
RAIL_TOLERANCE = 2.0
for (start, end) in RAILS:
    sx, sy = start
    ex, ey = end
    min_x = min(sx, ex) - RAIL_TOLERANCE
    max_x = max(sx, ex) + RAIL_TOLERANCE
    min_y = min(sy, ey) - RAIL_TOLERANCE
    max_y = max(sy, ey) + RAIL_TOLERANCE
    RAIL_BOUNDS.append((min_x, max_x, min_y, max_y))

def get_circular_roi(image, center, radius):
    # Extracts a rectangular Region of Interest (ROI) that encompasses a circle
    # around a given center point. Validates boundaries to ensure we don't try to
    # access pixels outside the image.
    #
    # Args:
    #     image (numpy.ndarray): The source image/frame.
    #     center (tuple): (x, y) coordinates of the center point.
    #     radius (int): The radius of the ROI to extract.
    #
    # Returns:
    #     tuple: (roi_image, (x1, y1, x2, y2)) or (None, None) if invalid.
    #            The second element is the bounding box coordinates in the original image.
    h, w = image.shape[:2]  # Get image height and width
    x, y = center

    # Calculate bounding box coordinates with boundary checks
    # Ensure x1/y1 are not less than 0, and x2/y2 do not exceed image width/height
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)

    # Validate that the resulting ROI has positive dimensions
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, None

    # Slice the image array to get the ROI
    roi = image[y1:y2, x1:x2]

    # Double check if the slice is empty
    if roi.size == 0:
        return None, None

    return roi, (x1, y1, x2, y2)

def detect_white_line(roi_hsv, roi_rect, ghostball_center, settings, white_lower, white_upper):
    # Detects the short white aim line within the Region of Interest (ROI).
    # Uses HSV color thresholding and the Hough Line Transform.
    #
    # Args:
    #     roi_hsv (numpy.ndarray): The ROI image converted to HSV color space.
    #     roi_rect (tuple): The (x1, y1, x2, y2) coordinates of the ROI in the full frame.
    #     ghostball_center (tuple): The (x, y) center of the detected ghostball.
    #     settings (dict): Dictionary containing detection parameters (HSV thresholds, etc).
    #     white_lower (numpy.ndarray): Pre-computed lower HSV bound for white color.
    #     white_upper (numpy.ndarray): Pre-computed upper HSV bound for white color.
    #
    # Returns:
    #     tuple: (x1, y1, x2, y2) coordinates of the best matching line in the *full frame*,
    #            or None if no valid line is found.

    # Create a binary mask where white pixels are 255 and others are 0
    # Uses the pre-computed lower and upper HSV bounds
    white_mask = cv2.inRange(roi_hsv, white_lower, white_upper)

    # Apply Probabilistic Hough Line Transform to find line segments
    # 1: distance resolution in pixels
    # np.pi/180: angle resolution in radians (1 degree)
    # line_threshold: minimum number of intersections to detect a line
    # minLineLength: minimum number of pixels making up a line
    # maxLineGap: maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(white_mask, 1, np.pi/180,
                           settings["line_threshold"],
                           minLineLength=settings["min_line_length"],
                           maxLineGap=settings["max_line_gap"])

    if lines is None:
        return None

    # We need to map the ROI coordinates back to the full frame coordinates
    x1_offset, y1_offset = roi_rect[0], roi_rect[1]

    min_dist_sq = float('inf')
    best_line = None
    max_dist_sq = MAX_ENDPOINT_DISTANCE ** 2  # Pre-compute squared threshold

    # Iterate through all detected lines to find the one originating from the ghostball
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Convert ROI-relative coordinates to absolute frame coordinates
        x1_frame = x1 + x1_offset
        y1_frame = y1 + y1_offset
        x2_frame = x2 + x1_offset
        y2_frame = y2 + y1_offset

        # Calculate squared distance from both endpoints to the ghostball center
        # Using squared distances avoids expensive sqrt operations for comparisons
        dist1_sq = (x1_frame - ghostball_center[0])**2 + (y1_frame - ghostball_center[1])**2
        dist2_sq = (x2_frame - ghostball_center[0])**2 + (y2_frame - ghostball_center[1])**2

        # We care about the endpoint closest to the ghostball
        min_endpoint_dist_sq = min(dist1_sq, dist2_sq)

        # Filter: Is the line touching (or very close to) the ghostball?
        # And is it the closest line we've found so far?
        if min_endpoint_dist_sq <= max_dist_sq and min_endpoint_dist_sq < min_dist_sq:
            min_dist_sq = min_endpoint_dist_sq
            best_line = (x1_frame, y1_frame, x2_frame, y2_frame)

    return best_line

def is_on_rail(point):
    # Checks if a given point lies on any of the defined rail segments.
    # Used to determine if the projected line has hit a rail or a pocket.
    # Uses pre-computed rail bounding boxes for performance.
    #
    # Args:
    #     point (tuple): (x, y) coordinates to check.
    #
    # Returns:
    #     bool: True if the point is considered to be on a rail, False otherwise.
    x, y = point

    # Check against pre-computed rail bounding boxes
    for min_x, max_x, min_y, max_y in RAIL_BOUNDS:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True

    return False

def extend_line_to_rails(line, ghostball_center):
    # Projects the detected aim line forward until it intersects with the table boundaries.
    # If the intersection is on a rail, it stops there. If it's in a gap (pocket),
    # it extends slightly further to indicate a pot.
    #
    # Args:
    #     line (tuple): The detected short line segment (x1, y1, x2, y2).
    #     ghostball_center (tuple): The origin of the shot.
    #
    # Returns:
    #     tuple: (start_x, start_y, end_x, end_y) representing the fully extended line.
    x1, y1, x2, y2 = line

    # Calculate the vector components of the line
    dx = x2 - x1
    dy = y2 - y1

    # Check for zero-length line to avoid division by zero
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return line

    # Normalize the vector to get direction (unit vector)
    length = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / length
    dy_norm = dy / length

    # Ensure the direction vector points AWAY from the ghostball
    # We check which endpoint is further from the ghostball and aim that way
    gb_x, gb_y = ghostball_center
    # Use squared distances to avoid expensive sqrt
    dist1_sq = (x1 - gb_x)**2 + (y1 - gb_y)**2
    dist2_sq = (x2 - gb_x)**2 + (y2 - gb_y)**2

    if dist2_sq < dist1_sq:
        # If the second point is closer, flip the direction
        dx_norm = -dx_norm
        dy_norm = -dy_norm

    # Use pre-computed table bounding box (no need to recalculate from RAILS)
    min_x, max_x = TABLE_MIN_X, TABLE_MAX_X
    min_y, max_y = TABLE_MIN_Y, TABLE_MAX_Y

    # Ray Casting: Find intersection with the four planes defining the table box
    # Parametric line equation: P = P0 + t * V
    # We solve for 't' (distance) where the line hits x=min_x, x=max_x, y=min_y, y=max_y
    t_values = []

    # Check X-planes (Left and Right walls)
    if abs(dx_norm) > 1e-6:
        t_left = (min_x - gb_x) / dx_norm
        t_right = (max_x - gb_x) / dx_norm
        # Only consider positive t (forward direction)
        if t_left > 0: t_values.append((t_left, 'x_plane'))
        if t_right > 0: t_values.append((t_right, 'x_plane'))

    # Check Y-planes (Top and Bottom walls)
    if abs(dy_norm) > 1e-6:
        t_top = (min_y - gb_y) / dy_norm
        t_bottom = (max_y - gb_y) / dy_norm
        # Only consider positive t (forward direction)
        if t_top > 0: t_values.append((t_top, 'y_plane'))
        if t_bottom > 0: t_values.append((t_bottom, 'y_plane'))

    # If no intersections found (unlikely unless inside-out or logic error), return original
    if not t_values:
        return (gb_x, gb_y, x2, y2)

    # The closest intersection (smallest positive t) is the one we hit first
    t_min, _ = min(t_values, key=lambda x: x[0])

    # Calculate the exact intersection point
    intersect_x = gb_x + t_min * dx_norm
    intersect_y = gb_y + t_min * dy_norm

    # Check if this intersection point lies on a physical rail
    if is_on_rail((intersect_x, intersect_y)):
        # It's a bounce/bank shot - stop exactly at the rail
        end_x, end_y = int(intersect_x), int(intersect_y)
    else:
        # If it's not on a rail, it must be in a gap (a pocket)
        # We extend the line slightly further (40px) to visualize the ball entering the pocket
        extension = 40.0
        end_x = int(intersect_x + extension * dx_norm)
        end_y = int(intersect_y + extension * dy_norm)

    return (gb_x, gb_y, end_x, end_y)

def load_settings():
    # Loads detection configuration from the JSON file.
    # Falls back to hardcoded defaults if the file is missing or unreadable.
    #
    # Returns:
    #     dict: The configuration dictionary.
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)

    print(f"Warning: {CONFIG_FILE} not found, using default settings")
    # Default settings calibrated for standard conditions
    return {
        "ghostball": {
            # HSV range for the pink ghostball
            "hsv_lower": [140, 50, 50],
            "hsv_upper": [170, 255, 255],
            "min_area": 200,      # Min contour area to be considered
            "max_area": 1000,     # Max contour area
            "min_circularity": 70 # Min circularity percentage
        },
        "white_line": {
            # HSV range for the white aim line
            "hsv_lower": [0, 0, 200],
            "hsv_upper": [180, 30, 255],
            "line_threshold": 30, # Hough threshold
            "min_line_length": 20,
            "max_line_gap": 10
        }
    }

def main():
    # Main execution loop.
    # Initializes the screen capture, processes frames, and handles user input.

    # Print welcome message and instructions
    print("=" * 60)
    print("BALLDETECTOR - PINK BALL & WHITE LINE DETECTION")
    print("=" * 60)
    print("Loading settings from", CONFIG_FILE)
    print("Press 'q' to quit")
    print("Press 'r' to toggle rails overlay")
    print("=" * 60)
    print()

    # Load settings
    settings = load_settings()
    print("Settings loaded successfully")
    print()

    # Pre-compute numpy arrays from settings (performance optimization)
    # These arrays are created once instead of every frame
    ghostball_lower = np.array(settings["ghostball"]["hsv_lower"])
    ghostball_upper = np.array(settings["ghostball"]["hsv_upper"])
    white_lower = np.array(settings["white_line"]["hsv_lower"])
    white_upper = np.array(settings["white_line"]["hsv_upper"])

    # Cache ghostball detection parameters
    gb_min_area = settings["ghostball"]["min_area"]
    gb_max_area = settings["ghostball"]["max_area"]
    gb_min_circularity = settings["ghostball"]["min_circularity"] / 100.0
    gb_early_exit_threshold = 0.85  # Exit early when we find a very circular object

    # Initialize MSS for screen capture
    # 'with' context manager ensures proper cleanup of resources
    with mss.mss() as sct:
        # Create an OpenCV window for display
        window_name = "Balldetector - Live Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # State variable for toggling the rail visualization
        show_rails = True

        while True:
            start_time = time.time()

            # 1. Capture the screen region defined by 'monitor'
            screenshot = sct.grab(monitor)

            # Convert the raw screenshot to a NumPy array for OpenCV
            frame = np.array(screenshot)

            # MSS returns BGRA, so convert to BGR (removing alpha channel)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Convert to HSV (Hue, Saturation, Value) for better color detection
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # Create a copy of the frame to draw overlays on (leaving original data pure)
            display_frame = frame_bgr.copy()

            # Draw rail boundaries if the toggle is enabled
            if show_rails:
                for p1, p2 in RAILS:
                    cv2.line(display_frame, p1, p2, (0, 0, 0), 1)

            # 2. Detect the Ghostball
            # Create a mask for the ghostball color using pre-computed bounds
            ghostball_mask = cv2.inRange(frame_hsv, ghostball_lower, ghostball_upper)

            # Apply rail boundary mask to only search within the play area
            # Create a rectangular mask for the table bounds
            play_area_mask = np.zeros(ghostball_mask.shape, dtype=np.uint8)
            cv2.rectangle(play_area_mask,
                         (TABLE_MIN_X, TABLE_MIN_Y),
                         (TABLE_MAX_X, TABLE_MAX_Y),
                         255, -1)  # -1 fills the rectangle

            # Combine ghostball mask with play area mask
            ghostball_mask = cv2.bitwise_and(ghostball_mask, play_area_mask)

            # Find contours (blobs) in the mask
            contours, _ = cv2.findContours(ghostball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ghostball_center = None

            # Iterate through contours to find the best match (highest circularity)
            # Using cached parameters for performance
            best_circularity = 0
            best_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)

                # Check area constraints using cached values
                if gb_min_area <= area <= gb_max_area:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Calculate circularity: 4*pi*area / perimeter^2
                        # A perfect circle has circularity of 1.0
                        circularity = 4 * np.pi * area / (perimeter ** 2)

                        # Check circularity constraint and track the best candidate
                        if circularity >= gb_min_circularity:
                            if circularity > best_circularity:
                                best_circularity = circularity
                                best_contour = contour

                                # Early exit: if we found a nearly perfect circle, no need to keep searching
                                if circularity > gb_early_exit_threshold:
                                    break

            # Draw and use the best matching ghostball
            if best_contour is not None:
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))

                # Draw detection marker (black circle) on display frame
                cv2.circle(display_frame, center, int(radius), (0, 0, 0), 1)
                cv2.circle(display_frame, center, 1, (0, 0, 0), 1)

                ghostball_center = center

            # 3. Detect and Extend the White Aim Line
            if ghostball_center is not None:
                # Get the ROI around the ghostball to search for the line
                roi_hsv, roi_rect = get_circular_roi(frame_hsv, ghostball_center, ROI_RADIUS)

                if roi_hsv is not None and roi_rect is not None:
                    # Detect the line within that ROI using pre-computed HSV arrays
                    line = detect_white_line(roi_hsv, roi_rect, ghostball_center,
                                            settings["white_line"], white_lower, white_upper)

                    if line is not None:
                        # Extend the detected line segment to the rails/pockets
                        extended_line = extend_line_to_rails(line, ghostball_center)
                        x1, y1, x2, y2 = extended_line

                        # Draw the extended aim line (Cyan color)
                        # BGR: (255, 255, 0)
                        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # 4. Show the result
            cv2.imshow(window_name, display_frame)

            # 5. Handle Loop Timing and Input
            elapsed_time = time.time() - start_time
            remaining_delay = FRAME_DELAY - elapsed_time

            # Wait for at least 1ms to allow OpenCV to process window events
            wait_time = max(1, int(remaining_delay * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                print("\nQuitting application...")
                break
            elif key == ord('r'):
                show_rails = not show_rails
                print(f"Rails overlay: {'On' if show_rails else 'Off'}")

        # Cleanup
        cv2.destroyAllWindows()
        print("Script terminated cleanly. All resources released.")

# Entry point check
if __name__ == "__main__":
    main()
