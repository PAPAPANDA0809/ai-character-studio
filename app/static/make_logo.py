"""Draws the Reelmint logo: a white film reel with a mint leaf on a mint-green tile."""
import math
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S = 4  # supersample for smooth edges


def logo(size, maskable=False):
    n = size * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(n):  # mint gradient
        t = y / n
        d.line([(0, y), (n, y)], fill=(int(46 + 30 * t), int(214 - 70 * t), int(170 - 40 * t), 255))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=0 if maskable else n // 5, fill=255)
    img.putalpha(mask)

    cx, cy = n / 2, n / 2
    R = n * (0.27 if maskable else 0.34)
    white = (255, 255, 255, 255)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=white)  # reel disc
    hole_r, ring = R * 0.22, R * 0.55
    for k in range(5):  # five sprocket holes
        a = -math.pi / 2 + k * 2 * math.pi / 5
        hx, hy = cx + ring * math.cos(a), cy + ring * math.sin(a)
        d.ellipse([hx - hole_r, hy - hole_r, hx + hole_r, hy + hole_r], fill=(40, 170, 140, 255))
    hub = R * 0.12
    d.ellipse([cx - hub, cy - hub, cx + hub, cy + hub], fill=(40, 170, 140, 255))
    # film tail sweeping to the bottom right
    d.rectangle([cx, cy + R - R * 0.18, cx + R * 1.25, cy + R], fill=white)
    # mint leaf tucked on the reel's top-right edge
    L = int(R * 0.95)
    leaf = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    ld.ellipse([0, L * 0.28, L, L * 0.72], fill=(214, 255, 238, 255), outline=(40, 170, 140, 255),
               width=max(3, n // 90))
    ld.line([(L * 0.08, L / 2), (L * 0.92, L / 2)], fill=(40, 170, 140, 255), width=max(3, n // 110))
    leaf = leaf.rotate(40, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(leaf, (int(cx + R * 0.45), int(cy - R * 1.2)))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    for s in (32, 192, 512):
        logo(s).save(os.path.join(HERE, f"reelmint-{s}.png"))
    logo(512, maskable=True).save(os.path.join(HERE, "reelmint-maskable-512.png"))
    print("logo written")
