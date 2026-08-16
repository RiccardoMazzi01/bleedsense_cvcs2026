import glob
import json
import os

RESULTS_DIR = "/work/cvcs2026/bleedsense/results"

rows = []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    with open(path) as f:
        data = json.load(f)

    hist = data.get("history", [])
    if not hist:
        continue

    epochs_trained = data.get("epochs_trained", len(hist))
    last_dice = hist[-1]["dice"]
    lookback = 6
    ref_idx = max(0, len(hist) - lookback)
    ref_dice = hist[ref_idx]["dice"]
    delta = last_dice - ref_dice

    hit_cap = epochs_trained >= 40
    rows.append((data["run_name"], epochs_trained, last_dice, delta, hit_cap))

print(f"{'run_name':45s} {'epochs':>7s} {'val_dice':>9s} {'delta_last6':>12s}  hit_40_cap")
for run_name, epochs_trained, last_dice, delta, hit_cap in rows:
    flag = " <-- CONTROLLARE" if hit_cap and delta > 0.01 else ""
    print(f"{run_name:45s} {epochs_trained:7d} {last_dice:9.4f} {delta:12.4f}  {str(hit_cap):>9s}{flag}")

still_improving = [r for r in rows if r[4] and r[3] > 0.01]
print(f"\nTotale run: {len(rows)}")
print(f"Run che hanno raggiunto il cap di 40 epoche E stavano ancora migliorando (delta ultime 6 > 0.01): {len(still_improving)}")
