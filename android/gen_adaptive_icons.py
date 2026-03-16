#!/usr/bin/env python3
"""Generate Android adaptive icon layers from icon-512.png (Menger W icon).

Android adaptive icons (API 26+) use two layers:
- Foreground: icon content centered in a 108dp canvas, safe zone = inner 66%
- Background: solid color or pattern behind the foreground

The OS applies a device-specific mask (circle, squircle, rounded square).
Content outside the safe zone may be clipped.

Also generates legacy launcher icons at all mipmap densities.
"""

from PIL import Image
import os
import shutil

SOURCE = "/home/solaya/Desktop/TOE/hive/icon-512.png"
ANDROID_DIR = "/home/solaya/Desktop/TOE/hive/android"

# Background color sampled from the icon's dark background
BG_COLOR = (17, 17, 24, 255)  # ~#111118

# Adaptive icon canvas is 108dp. Foreground safe zone is inner 72dp (66.67%).
# Padding on each side = (108 - 72) / 2 / 108 = 16.67% of canvas
SAFE_ZONE_RATIO = 72 / 108  # 0.6667

# Output size for the high-res adaptive layers (xxxhdpi = 432px for 108dp)
ADAPTIVE_SIZE = 512  # We'll use 512 for quality, buildozer scales down


def generate_foreground(src_path, out_path, size=ADAPTIVE_SIZE):
    """Place icon content centered within adaptive icon safe zone."""
    src = Image.open(src_path).convert("RGBA")

    # Create transparent canvas
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Scale icon to fit within safe zone
    safe_px = int(size * SAFE_ZONE_RATIO)
    icon_resized = src.resize((safe_px, safe_px), Image.LANCZOS)

    # Center it
    offset = (size - safe_px) // 2
    canvas.paste(icon_resized, (offset, offset), icon_resized)

    canvas.save(out_path, "PNG")
    print(f"  Foreground: {out_path} ({size}x{size}, safe zone {safe_px}px)")


def generate_background(out_path, size=ADAPTIVE_SIZE, color=BG_COLOR):
    """Solid dark background layer."""
    img = Image.new("RGBA", (size, size), color)
    img.save(out_path, "PNG")
    print(f"  Background: {out_path} ({size}x{size}, color #{color[0]:02x}{color[1]:02x}{color[2]:02x})")


def generate_legacy_icons(src_path, out_dir):
    """Generate legacy launcher icons at Android mipmap sizes."""
    # Standard Android launcher icon sizes per density
    sizes = {
        "mdpi": 48,
        "hdpi": 72,
        "xhdpi": 96,
        "xxhdpi": 144,
        "xxxhdpi": 192,
    }
    src = Image.open(src_path).convert("RGBA")

    os.makedirs(out_dir, exist_ok=True)
    for density, px in sizes.items():
        icon = src.resize((px, px), Image.LANCZOS)
        path = os.path.join(out_dir, f"ic_launcher_{density}_{px}x{px}.png")
        icon.save(path, "PNG")
        print(f"  Legacy {density:8s}: {path} ({px}x{px})")


if __name__ == "__main__":
    print("Generating Android adaptive icon layers...\n")

    fg_path = os.path.join(ANDROID_DIR, "icon_adaptive_fg.png")
    bg_path = os.path.join(ANDROID_DIR, "icon_adaptive_bg.png")

    generate_foreground(SOURCE, fg_path)
    generate_background(bg_path)

    print("\nGenerating legacy launcher icons...\n")
    legacy_dir = os.path.join(ANDROID_DIR, "launcher_icons")
    generate_legacy_icons(SOURCE, legacy_dir)

    # Copy the W icon as the main android icon (replacing the non-W version)
    main_icon = os.path.join(ANDROID_DIR, "icon-512.png")
    shutil.copy2(SOURCE, main_icon)
    print(f"\n  Copied W icon to {main_icon}")

    print("\nDone! Update buildozer.spec with:")
    print("  icon.adaptive_foreground.filename = %(source.dir)s/icon_adaptive_fg.png")
    print("  icon.adaptive_background.filename = %(source.dir)s/icon_adaptive_bg.png")
