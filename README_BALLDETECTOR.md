# Balldetector.py - Pink Ball & White Line Detection Calibration Tool

## Overview

`balldetector.py` is a calibration tool for detecting the pink ghostball and white aim line in billiards gameplay. It provides real-time HSV color filtering with trackbars for fine-tuning detection parameters.

## Features

### Dual Detection System
1. **Ghostball Detection (Pink Ball)** - Uses contour detection with circularity filtering
2. **White Line Detection** - Uses HoughLinesP within a constrained ROI around the ghostball

### Display Modes
- **Live View** - Shows original feed with detection overlays
- **Mask View** - Shows binary mask (white = detected regions)
- **Outline View** - Shows yellow contours around detected regions

### Real-time Calibration
- Separate trackbars for ghostball and white line parameters
- Settings auto-save/load from `line_detection_settings.json`
- Settings reload when switching between masks

## Keyboard Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Ghostball mask (pink ball detection) |
| `2` | Switch to White Line mask (line detection) |
| `m` | Cycle display modes (Live → Mask → Outline) |
| `s` | Save ONLY current active mask settings to JSON |
| `q` / `Q` / `ESC` | Quit application |

## Trackbar Parameters

### Ghostball (GB) Parameters
- **GB H Low/High** - Hue range for pink color (0-180)
- **GB S Low/High** - Saturation range (0-255)
- **GB V Low/High** - Value/brightness range (0-255)
- **GB Min Area** - Minimum contour area in pixels (200-2000)
- **GB Max Area** - Maximum contour area in pixels (200-2000)
- **GB Circularity** - Minimum circularity percentage (70-100)

### White Line (WL) Parameters
- **WL H Low/High** - Hue range for white color (0-180)
- **WL S Low/High** - Saturation range (0-255)
- **WL V Low/High** - Value/brightness range (0-255)
- **WL Threshold** - HoughLinesP accumulator threshold (0-100)
- **WL Min Length** - Minimum line length in pixels (0-100)
- **WL Max Gap** - Maximum gap between line segments (0-50)

## Calibration Workflow

### Step 1: Calibrate Ghostball
1. Run `python balldetector.py`
2. Press `1` to activate Ghostball mask
3. Adjust GB trackbars until only the pink ball is detected
4. Press `m` to view Mask mode (verify white region matches pink ball)
5. Press `s` to save ghostball settings

### Step 2: Calibrate White Line
1. Press `2` to activate White Line mask
2. **IMPORTANT**: Ghostball must be detected first (line detection uses ghostball position)
3. Adjust WL trackbars until white aim line is detected
4. Press `m` to view Mask mode (verify white region matches aim line)
5. Press `s` to save white line settings

## Detection Details

### Ghostball Detection Method
- Converts frame to HSV color space
- Creates binary mask using HSV ranges
- Finds contours in mask
- Filters by area: `min_area <= area <= max_area`
- Filters by circularity: `circularity >= min_circularity`
- Circularity formula: `4π × area / perimeter²`
- Draws black circle overlay on detected ball

### White Line Detection Method
- Requires ghostball to be detected first
- Searches within 70px circular ROI around ghostball center
- Uses HoughLinesP for line detection
- Filters lines by endpoint proximity to ghostball (< 5px)
- Extends detected line from ghostball center to frame edge in ONE direction
- Direction determined by which line endpoint is farther from ghostball
- Draws cyan (1px) line overlay

### Line Extension Algorithm
```
1. Calculate direction vector from detected line segment
2. Normalize direction to unit length
3. Determine which line endpoint is closer to ghostball
4. Reverse direction if needed (point away from closer endpoint)
5. Find intersection with frame edges using parametric equations
6. Extend from ghostball center to first frame edge hit
```

## Configuration

### Monitor Region
Located at top of script:
```python
monitor = {"left": 2151, "top": 75, "width": 1380, "height": 750}
```
**IMPORTANT**: Update these coordinates to match your actual pool table position on screen.

### ROI and Filtering Constants
```python
ROI_RADIUS = 70              # Search radius around ghostball for white line
MAX_ENDPOINT_DISTANCE = 5    # Max distance from ghostball to line endpoint
```

### Settings File
- **Location**: `line_detection_settings.json` (auto-created in script directory)
- **Format**: JSON with separate sections for ghostball and white_line
- **Auto-load**: Settings reload from file when switching masks (press 1 or 2)
- **Auto-save**: Only saves current active mask when pressing 's'

## Windows Layout

### Two-Window System
1. **"Controls" window** (resizable) - Contains all trackbars
2. **"Pink Ball & Line Detection" window** (fixed size) - Shows live feed/mask/outline

## Terminal Output

### Startup Banner
```
============================================================
BILLIARDS LINE DETECTION - CALIBRATION TOOL
============================================================

WINDOWS:
  • 'Controls' window (resizable) - Adjust trackbars here
  • 'Pink Ball & Line Detection' - View live feed (fixed size)

KEYBOARD CONTROLS:
  [1] - Switch to Ghostball mask (pink ball detection)
  [2] - Switch to White Line mask (line detection)
  ...
```

### Key Press Feedback
All key presses print confirmation to terminal:
```
[KEY PRESS] '1' - Switching to Ghostball mask
  → Ghostball mask active (pink ball detection)
  → Loaded HSV: [[140, 50, 50]] to [[170, 255, 255]]
```

### Save Confirmation
Displays all saved parameter values:
```
[KEY PRESS] 'S' - Saving settings
  → Saved Ghostball settings:
     HSV: [140, 50, 50] to [170, 255, 255]
     Area: 200 to 1000, Circularity: 70%
```

## Troubleshooting

### No ghostball detected
- Check monitor coordinates match actual table position on screen
- Use Mask view (`m` key) to verify HSV range captures pink ball
- Adjust Min/Max Area trackbars if ball size is different
- Lower Circularity threshold if ball appears oval/distorted

### No white line detected
- Ensure ghostball is detected first (line detection requires ghostball position)
- Verify ghostball center is within 70px of the white line
- Use Mask view to verify white line appears in binary mask
- Adjust WL Threshold, Min Length, and Max Gap trackbars

### Line points wrong direction
- Line should extend FROM ghostball AWAY FROM the cue ball
- If reversed, the detected line endpoints may be in wrong order
- Try adjusting detection parameters to get cleaner line detection

### Settings not persisting
- Verify you pressed 's' key while correct mask is active (1 or 2)
- Check `line_detection_settings.json` exists and is valid JSON
- Settings are per-mask; switching masks loads different settings
- Each mask's settings are independent

### Can't quit application
- Try pressing `q`, `Q`, or `ESC` key
- Make sure OpenCV window has focus (click on it)
- Check terminal for "[KEY PRESS]" messages to verify key detection

## Technical Details

### Dependencies
```python
import cv2              # OpenCV for image processing
import numpy as np      # Array operations
import mss              # Fast screen capture
import time             # FPS control
import json             # Settings persistence
import os               # File operations
```

### Frame Rate
- Target: 45 FPS
- Frame delay: ~22ms between frames
- Uses `cv2.waitKey()` for timing and keyboard input

### Color Spaces
- **Input**: BGRA from screen capture
- **Processing**: HSV for color filtering
- **Display**: BGR for visualization

## Default Settings

### Ghostball Defaults
```json
{
  "hsv_lower": [140, 50, 50],
  "hsv_upper": [170, 255, 255],
  "min_area": 200,
  "max_area": 1000,
  "min_circularity": 70
}
```

### White Line Defaults
```json
{
  "hsv_lower": [0, 0, 200],
  "hsv_upper": [180, 30, 255],
  "line_threshold": 30,
  "min_line_length": 20,
  "max_line_gap": 10
}
```



## Notes

- Detection overlays: Ghostball = black circle, White line = cyan line
- Settings are saved per-mask, not globally
- Line extension uses parametric line equations for accurate frame edge intersection
- ROI (Region of Interest) validation prevents crashes when ghostball is near frame edges

## Version History

- **Current**: Multi-mask calibration with auto-reload, one-directional line extension
- Improved keyboard feedback with detailed terminal output
- Added ESC key support for quitting
- Fixed ROI validation to prevent crashes
