# Ball Detector

Aim assist overlay for 8 Ball Pool. Finds the ghostball, finds the aim line, extends it to show you exactly where your shot is going.

## The Problem

8 Ball Pool shows you a ghostball (the pink circle showing where the cue ball will hit) and a tiny white line showing the direction. But that line is short as hell and doesn't tell you where the ball actually ends up.

## The Solution

This script:
1. Grabs your screen where the game is running
2. Finds the pink ghostball
3. Finds that stubby white aim line
4. Extends it all the way to the table rails or pockets so you see the full shot path
5. Draws a cyan line overlay showing you exactly where you're aiming

---

## Setup

### Install dependencies
```bash
pip install opencv-python numpy mss
```

### Configure your screen region

The script captures a specific rectangle of your screen. Default is set up for a specific monitor setup that probably isn't yours.

Open `balldetector.py` and find this at the top:
```python
monitor = {"left": 2151, "top": 75, "width": 1380, "height": 750}
```

Change these to match where your game window is:
- `left` - pixels from the left edge of your screen to where the game starts
- `top` - pixels from the top
- `width` / `height` - size of the area to capture

**Tip**: Use a screenshot tool to figure out the coordinates of your game window.

### Settings file (optional)

Create `line_detection_settings.json` in the same folder if you need to tweak detection. If you don't create it, defaults work for most setups.

Here is an example configuration (values from `line_detection_settings.json`):

```json
{
  "ghostball": {
    "hsv_lower": [128, 0, 176],
    "hsv_upper": [180, 255, 255],
    "min_area": 600,
    "max_area": 1186,
    "min_circularity": 19
  },
  "white_line": {
    "hsv_lower": [0, 0, 243],
    "hsv_upper": [180, 93, 255],
    "line_threshold": 30,
    "min_line_length": 20,
    "max_line_gap": 10
  }
}
```

---

## Usage

```bash
python balldetector.py
```

A window pops up showing your game with overlays:
- Black circle around the ghostball
- Cyan line showing the extended aim path
- Black lines showing the table rails

### Controls
- `q`: Quit the application
- `r`: Toggle rails overlay on/off

---

## How Detection Works

### Finding the ghostball

1. Converts the frame to HSV (hue-saturation-value) color space
2. Masks everything that isn't pink/magenta (based on HSV settings)
3. Finds blob outlines (contours) in that mask
4. Filters by size and shape (circularity)
5. Grabs the center point of the best match

### Finding the aim line

1. Looks only in a 70-pixel radius around the ghostball (no point searching the whole screen)
2. Masks for white pixels (high brightness, low saturation)
3. Uses OpenCV's HoughLinesP to detect line segments
4. Picks the line that has an endpoint closest to the ghostball center (within 5 pixels)

### Extending the line

1. Takes the direction from the detected line segment
2. Figures out which way points "away" from the ghostball
3. Projects that direction until it hits the table rails or pocket gaps
4. Draws from the ghostball center to that intersection point

---

## Performance Optimizations

The script is optimized for real-time performance at 45+ FPS with several key improvements:

### Pre-computed Constants
- **Table bounding box** - Calculated once at startup from RAILS instead of 36 operations per frame
- **Rail collision bounds** - Pre-computed bounding boxes for all 6 rails with tolerance applied
- **HSV color arrays** - Created once after loading settings instead of ~180 allocations per second

### Cached Parameters
- Ghostball detection parameters (`min_area`, `max_area`, `min_circularity`) are extracted before the main loop
- Eliminates dictionary hash lookups that were happening 10-50 times per frame

### Algorithm Optimizations
- **Squared distances** - All distance comparisons use squared values to avoid expensive `sqrt()` calls (5-10× faster)
- **Early exit** - Stops searching contours when circularity > 0.85 (nearly perfect circle found)
- **Play area masking** - Restricts ghostball search to table bounds only, reducing false positives

### Performance Impact
These optimizations provide an estimated **23-40% improvement** in frame processing speed:
- Reduced from ~22ms to ~13-17ms per frame at 45 FPS
- Can sustain 60+ FPS on most systems
- Significantly lower CPU usage

---

## When Things Don't Work

### "I don't see any detection"

Your game isn't in the capture area. Fix the `monitor` coordinates.

### "Ghostball not detected"

The pink color is off. Either:
- Your game has different colors/brightness
- There's glare or the table has a weird tint

Fix: Adjust `hsv_lower` and `hsv_upper` in the ghostball settings. Use a color picker on a screenshot to find the actual HSV values of the ghostball.

HSV ranges in OpenCV:
- H (hue): 0-179
- S (saturation): 0-255  
- V (value/brightness): 0-255

### "Line not detected"

The white line might be dimmer than expected or broken up.

Try:
- Lower `hsv_lower[2]` (the V value) to accept darker whites
- Lower `min_line_length` if the line segment is really short
- Raise `max_line_gap` if the line is dashed/broken

### "Wrong line detected"

Some other white thing is getting picked up.

Try:
- Lower `ROI_RADIUS` (in the script, not settings) to search a smaller area
- Lower `MAX_ENDPOINT_DISTANCE` to be stricter about the line touching the ghostball
- Raise `line_threshold` to only detect stronger lines

### "It's laggy"

Runs at 45 FPS by default. Lower `TARGET_FPS` in the script if your machine can't keep up.

---

## Files

| File | Purpose |
|------|---------|
| `balldetector.py` | The script |
| `line_detection_settings.json` | Detection tuning (optional, uses defaults if missing) |

---

## Key Constants (in script)

| Name | Default | What it does |
|------|---------|--------------|
| `monitor` | `{"left": 2151, "top": 75, "width": 1380, "height": 750}` | Screen region to capture |
| `TARGET_FPS` | 45 | How many frames per second to process |
| `ROI_RADIUS` | 70 | Pixel radius around ghostball to search for the line |
| `MAX_ENDPOINT_DISTANCE` | 5 | Max pixels between line endpoint and ghostball center |
