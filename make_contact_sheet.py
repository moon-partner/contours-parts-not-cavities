"""Contact sheet: tile all ABO front-view previews into one labeled grid image."""
import glob
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "real", "abo", "rendered")
OUT = os.path.join(HERE, "real", "abo", "contact_sheet.png")
CELL = 256
PAD = 4
LABEL = 22

files = sorted(glob.glob(os.path.join(SRC, "*", "*_front.png")))
if not files:
    raise SystemExit("no previews found -- render first")
cols = min(5, len(files))
rows = (len(files) + cols - 1) // cols
W = cols * (CELL + PAD) + PAD
H = rows * (CELL + LABEL + PAD) + PAD
sheet = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(sheet)

for i, f in enumerate(files):
    name = os.path.basename(os.path.dirname(f))
    im = Image.open(f).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    c, r = i % cols, i // cols
    x = PAD + c * (CELL + PAD)
    y = PAD + r * (CELL + LABEL + PAD)
    sheet.paste(im, (x, y))
    draw.text((x + 3, y + CELL + 3), name, fill=(0, 0, 0))

sheet.save(OUT)
print(f"wrote {OUT}  ({len(files)} previews, {cols}x{rows})")
