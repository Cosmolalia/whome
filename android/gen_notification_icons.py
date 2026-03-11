#!/usr/bin/env python3
"""Generate Menger L1 notification icons for Android.

Menger L1 = 3x3 grid with center removed = 8 white squares on transparent.
Android notification small icons must be white on transparent (monochrome).
"""

from PIL import Image, ImageDraw
import os

OUTPUT_DIR = "/home/solaya/Desktop/TOE/hive/android/notification_icons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Android density buckets: name -> pixel size
SIZES = {
    "mdpi":    24,
    "hdpi":    36,
    "xhdpi":   48,
    "xxhdpi":  72,
    "xxxhdpi": 96,
}

# Menger L1 mask: 3x3 grid, center (1,1) removed
CELLS = [
    (0, 0), (1, 0), (2, 0),
    (0, 1),         (2, 1),
    (0, 2), (1, 2), (2, 2),
]


def generate_icon(size, name):
    """Generate a single Menger L1 icon at the given pixel size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gap scales with resolution
    if size <= 24:
        gap = 1
    elif size <= 48:
        gap = 2
    else:
        gap = 3

    # Total gap space: 2 internal gaps (between 3 columns/rows)
    # Plus 0 outer padding — icon fills the full canvas
    total_gap = gap * 2
    cell_size = (size - total_gap) // 3

    # Recompute actual used size to center within canvas
    actual = cell_size * 3 + gap * 2
    offset = (size - actual) // 2

    for (cx, cy) in CELLS:
        x0 = offset + cx * (cell_size + gap)
        y0 = offset + cy * (cell_size + gap)
        x1 = x0 + cell_size - 1
        y1 = y0 + cell_size - 1
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 255))

    path = os.path.join(OUTPUT_DIR, f"ic_notification_menger_{name}_{size}x{size}.png")
    img.save(path, "PNG")
    print(f"  {name:8s}  {size:3d}x{size:<3d}  -> {path}")
    return path


if __name__ == "__main__":
    print("Generating Menger L1 notification icons...\n")
    for name, size in SIZES.items():
        generate_icon(size, name)
    print(f"\nDone. Icons saved to {OUTPUT_DIR}/")
