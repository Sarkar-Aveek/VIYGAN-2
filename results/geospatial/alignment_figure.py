"""Picture + flicker GIF: s2colour output vs its 8-date input, one 4x4-tile chip per test place."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")      # needs code/data/test (the test set) to run
sys.path.insert(0, CODE)
os.chdir(CODE)
import bootstrap  # noqa
import numpy as np, torch, cv2
import torch.nn.functional as F
from PIL import Image, ImageDraw
import ps_proof as pp
import infer


def our_model(name):
    """Our two models from ../../weights; bicubic from evaluate.py."""
    if name == "bicubic_8date":
        return pp.ours(name)
    net = infer.load(name, os.path.join("..", "weights", f"{name}_generator.pth"), torch.device("cuda"))
    return net

OUT = HERE
dev = torch.device("cuda")
fn = our_model("s2colour")
places = pp.test_places("inseason")


def edges(img, sigma):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), sigma)
    return cv2.Canny(g, 20, 60) > 0


rows, gif_pairs = [], []
for place, tiles in places.items():
    best = None
    for cx, cy, members in pp.chips(tiles):
        if len(members) != 16:
            continue
        hr = np.zeros((512, 512, 3), np.uint8)
        for (i, j), t in members.items():
            hr[j * 128:(j + 1) * 128, i * 128:(i + 1) * 128] = tiles[t]["hr"]
        s = edges(hr, 3).mean()
        if best is None or s > best[0]:
            best = (s, members, hr)
    _, members, hr = best
    ids = list(members.values())
    with torch.no_grad():
        x = torch.stack([tiles[t]["lr8"] for t in ids]).to(dev).float() / 255
        sr = fn(x).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        med = x.view(-1, 8, 3, 32, 32).median(1).values
        up = F.interpolate(med, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        up = up.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    A, B = np.zeros_like(hr), np.zeros_like(hr)
    for k, ((i, j), t) in enumerate(members.items()):
        A[j * 128:(j + 1) * 128, i * 128:(i + 1) * 128] = up[ids.index(t)]
        B[j * 128:(j + 1) * 128, i * 128:(i + 1) * 128] = sr[ids.index(t)]
    # overlay on coarse structure (sigma 4 HR px ~ 1 LR px): magenta = input, green = output, white = both
    ea, eb = cv2.dilate(edges(A, 4).astype(np.uint8), None), cv2.dilate(edges(B, 4).astype(np.uint8), None)
    ov = (np.stack([cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)] * 3, -1) * 0.35).astype(np.uint8)
    ov[ea > 0] = (255, 0, 255)
    ov[eb > 0] = (0, 255, 0)
    ov[(ea > 0) & (eb > 0)] = (255, 255, 255)
    rows.append((place, [A, B, hr, ov]))
    gif_pairs.append((place, A, B))

# sheet
titles = ["input: 8-date median (bicubic x4)", "s2colour output (2.39 m)", "ArcGIS reference (not a model input)",
          "edges: input magenta / output green / both white"]
S, pad, head = 384, 6, 22
W = 4 * S + 5 * pad + 150
H = head + len(rows) * (S + pad) + pad
sheet = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(sheet)
for c, t in enumerate(titles):
    d.text((150 + pad + c * (S + pad), 5), t, fill=(0, 0, 0))
for r, (place, ims) in enumerate(rows):
    y = head + r * (S + pad)
    d.text((6, y + S // 2), place, fill=(0, 0, 0))
    for c, im in enumerate(ims):
        sheet.paste(Image.fromarray(im).resize((S, S), Image.LANCZOS), (150 + pad + c * (S + pad), y))
sheet.save(os.path.join(OUT, "alignment_s2colour.png"))

# flicker GIF: input <-> output, 3 places side by side, with a fixed grid to judge position against
frames = []
pick = [p for p in gif_pairs if p[0] in ("hyderabad_charminar", "varanasi_ghats", "thanjavur_paddy")] or gif_pairs[:3]
for which, label in ((1, "INPUT (8-date median)"), (2, "s2colour OUTPUT")):
    canvas = Image.new("RGB", (3 * 512 + 4 * 8, 512 + 40), (255, 255, 255))
    dd = ImageDraw.Draw(canvas)
    dd.text((8, 12), label + "  - fixed grid lines every 64 px; nothing should jump", fill=(0, 0, 0))
    for k, (place, A, B) in enumerate(pick):
        im = Image.fromarray(A if which == 1 else B)
        g = ImageDraw.Draw(im)
        for v in range(64, 512, 64):
            g.line([(v, 0), (v, 511)], fill=(255, 255, 0))
            g.line([(0, v), (511, v)], fill=(255, 255, 0))
        canvas.paste(im, (8 + k * (512 + 8), 36))
    frames.append(canvas)
frames[0].save(os.path.join(OUT, "alignment_s2colour_flicker.gif"), save_all=True, append_images=frames[1:], duration=700, loop=0)
print("done")
