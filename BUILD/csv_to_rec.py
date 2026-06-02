import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_CSV_DIR = r"..."
BASE_OUT_DIR = r"..."
RP_SIZE      = 224
THRESHOLD    = 0.20
MIN_PKTS     = 10
MAX_PKTS     = 1000   

def parse_csv(csv_path):
    sessions = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(',')
            try:
                label    = cols[0].strip().lower()
                num_pkts = int(cols[7])
                if num_pkts < MIN_PKTS:
                    continue
                ts    = np.array([float(x) for x in cols[8:8 + num_pkts]], dtype=np.float32)
                sizes = np.array([int(x) for x in cols[9 + num_pkts:9 + num_pkts * 2] if x.strip()], dtype=np.float32)
                if len(ts) >= MIN_PKTS and len(sizes) >= MIN_PKTS:
                    sessions.append((ts, sizes))
            except (ValueError, IndexError):
                continue
    return sessions

def compute_rp(ts, sizes):
    n = min(len(ts), len(sizes), MAX_PKTS)
    ts, sizes = ts[:n], sizes[:n]
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8) # stability
    X    = np.stack([norm(ts), norm(sizes)], axis=1)
    diff = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    diff = diff / (diff.max() + 1e-8)
    return (diff <= THRESHOLD).astype(np.uint8)

def save_png(rp, path):
    n   = rp.shape[0]
    idx = (np.arange(RP_SIZE) * n / RP_SIZE).astype(int)
    rp  = rp[np.ix_(idx, idx)]
    img = np.where(rp > 0, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

total_saved = 0

for cls in sorted(os.listdir(BASE_CSV_DIR)):
    cls_path = os.path.join(BASE_CSV_DIR, cls)
    if not os.path.isdir(cls_path) or cls.startswith('.'):
        continue

    for variant in sorted(os.listdir(cls_path)):
        variant_path = os.path.join(cls_path, variant)
        if not os.path.isdir(variant_path) or variant.startswith('.'):
            continue

        csv_files = [f for f in os.listdir(variant_path) if f.endswith('.csv')]
        if not csv_files:
            continue

        out_dir = os.path.join(BASE_OUT_DIR, cls, variant)
        os.makedirs(out_dir, exist_ok=True)

        for csv_file in csv_files:
            csv_path = os.path.join(variant_path, csv_file)
            sessions = parse_csv(csv_path)
            print(f"[{cls}/{variant}] {csv_file} → {len(sessions)} sessions")

            for i, (ts, sizes) in enumerate(sessions):
                rp   = compute_rp(ts, sizes)
                path = os.path.join(out_dir, f"{cls}_{variant}_{i:06d}.png")
                save_png(rp, path)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1} saved...")

            total_saved += len(sessions)
            print(f" {len(sessions)} images saved to {out_dir}")