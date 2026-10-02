"""Draws the MyApp launcher icons (run with any Python that has Pillow)."""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def icon(size, maskable=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = 0 if maskable else size // 5
    # gradient background
    for y in range(size):
        t = y / size
        col = (int(92 + 40 * t), int(110 - 30 * t), int(255 - 60 * t), 255)
        d.line([(0, y), (size, y)], fill=col)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    img.putalpha(mask)
    # 2x2 tile grid
    pad = size * (0.30 if maskable else 0.22)
    gap = size * 0.06
    cell = (size - 2 * pad - gap) / 2
    for i in range(2):
        for j in range(2):
            x0 = pad + j * (cell + gap)
            y0 = pad + i * (cell + gap)
            d.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=cell * 0.25,
                                fill=(255, 255, 255, 235 if (i + j) % 2 == 0 else 160))
    return img


for s in (192, 512):
    icon(s).save(os.path.join(HERE, f"icon-{s}.png"))
icon(512, maskable=True).save(os.path.join(HERE, "icon-maskable-512.png"))
print("icons written")
