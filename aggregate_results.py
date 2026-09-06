import json
import os
import statistics as stats

RESULTS_DIR = "/work/cvcs2026/bleedsense/results"
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
AUGMENTATIONS = ["light", "aggressive"]
ADAPTATIONS = ["none", "reinhard", "fda"]
BLOOD_INDEX_OPTIONS = [False, True]
MIXSTYLE_OPTIONS = [False, True]
METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95"]


def suffix(augmentation, adaptation, use_blood_index=False, mixstyle=False):
    s = "" if augmentation == "light" else f"_aug{augmentation}"
    s += "" if adaptation == "none" else f"_adapt{adaptation}"
    s += "_bloodindex" if use_blood_index else ""
    s += "_mixstyle" if mixstyle else ""
    return s


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_std(values):
    m = stats.mean(values)
    s = stats.stdev(values) if len(values) > 1 else 0.0
    return m, s


def fmt(values):
    if not values:
        return "-"
    m, s = mean_std(values)
    if len(values) > 1:
        return f"{m:.3f}±{s:.3f}"
    return f"{m:.3f}"


def summarize_hemoset(arch, augmentation, adaptation, use_blood_index=False, mixstyle=False):
    runs = [load(f"hemoset_{arch}_fold{i}{suffix(augmentation, adaptation, use_blood_index, mixstyle)}")
            for i in range(5)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None, None
    indomain = {m: [r["indomain_test"][m] for r in runs] for m in METRICS}
    cross = {m: [r["cross_dataset_test"][m] for r in runs] for m in METRICS}
    return indomain, cross


def summarize_rabbani(arch, augmentation, adaptation, use_blood_index=False, mixstyle=False):
    run = load(f"rabbani_{arch}{suffix(augmentation, adaptation, use_blood_index, mixstyle)}")
    if run is None:
        return None, None
    indomain = {m: [run["indomain_test"][m]] for m in METRICS}
    cross = {m: [run["cross_dataset_test"][m]] for m in METRICS}
    return indomain, cross


def main():
    rows = []
    for arch in ARCHITECTURES:
        for aug in AUGMENTATIONS:
            for adapt in ADAPTATIONS:
                if aug != "light" and adapt != "none":
                    continue  # combinazione non eseguita in nessuna fase per ora

                for use_bi in BLOOD_INDEX_OPTIONS:
                    if use_bi and not (aug == "aggressive" and adapt == "none"):
                        continue  # blood-index testato solo sopra la condizione migliore finora

                    for use_ms in MIXSTYLE_OPTIONS:
                        if use_ms and not (aug == "light" and adapt == "none" and not use_bi):
                            continue  # mixstyle testato solo su light/none, isolato dagli altri assi
                        if use_bi and use_ms:
                            continue  # combinazione non eseguita

                        h_in, h_cross = summarize_hemoset(arch, aug, adapt, use_bi, use_ms)
                        r_in, r_cross = summarize_rabbani(arch, aug, adapt, use_bi, use_ms)
                        if h_in is not None:
                            rows.append(("HemoSet -> HemoSet (in-domain)", arch, aug, adapt, use_bi, use_ms, h_in))
                            rows.append(("HemoSet -> Rabbani (cross)", arch, aug, adapt, use_bi, use_ms, h_cross))
                        if r_in is not None:
                            rows.append(("Rabbani -> Rabbani (in-domain)", arch, aug, adapt, use_bi, use_ms, r_in))
                            rows.append(("Rabbani -> HemoSet (cross)", arch, aug, adapt, use_bi, use_ms, r_cross))

    if not rows:
        print("Nessun risultato trovato in", RESULTS_DIR)
        return

    header = ("| Setting | Architettura | Augmentation | Adaptation | BloodIndex | MixStyle | "
              + " | ".join(m.upper() for m in METRICS) + " |")
    sep = "|---" * (6 + len(METRICS)) + "|"
    lines = [header, sep]
    for setting, arch, aug, adapt, use_bi, use_ms, metrics in rows:
        cells = " | ".join(fmt(metrics[m]) for m in METRICS)
        lines.append(f"| {setting} | {arch} | {aug} | {adapt} | {use_bi} | {use_ms} | {cells} |")

    table_md = "\n".join(lines)
    print(table_md)

    out_path = os.path.join(RESULTS_DIR, "results_summary.md")
    with open(out_path, "w") as f:
        f.write("# BleedSense - Riepilogo risultati (in-domain vs cross-dataset)\n\n")
        f.write("Media +/- deviazione standard sui 5 fold per HemoSet; run singolo per Rabbani.\n")
        f.write("Righe assenti se quella combinazione non e' ancora stata eseguita.\n\n")
        f.write(table_md + "\n")

    print(f"\nSalvato anche in {out_path}")


if __name__ == "__main__":
    main()
