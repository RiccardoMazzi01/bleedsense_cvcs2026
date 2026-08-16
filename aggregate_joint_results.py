import json
import os
import statistics as stats

RESULTS_DIR = "/work/cvcs2026/bleedsense/results"
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95"]


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


def hemoset_baseline(arch):
    runs = [load(f"hemoset_{arch}_fold{i}") for i in range(5)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    return {m: [r["indomain_test"][m] for r in runs] for m in METRICS}


def rabbani_baseline(arch):
    run = load(f"rabbani_{arch}")
    if run is None:
        return None
    return {m: [run["indomain_test"][m]] for m in METRICS}


def joint_metrics(arch):
    runs = [load(f"joint_{arch}_fold{i}") for i in range(5)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None, None
    hemoset = {m: [r["hemoset_test"][m] for r in runs] for m in METRICS}
    rabbani = {m: [r["rabbani_test"][m] for r in runs] for m in METRICS}
    return hemoset, rabbani


def main():
    rows = []
    for arch in ARCHITECTURES:
        h_base = hemoset_baseline(arch)
        r_base = rabbani_baseline(arch)
        h_joint, r_joint = joint_metrics(arch)

        if h_base is not None:
            rows.append(("HemoSet test", arch, "single-source (Fase 1)", h_base))
        if h_joint is not None:
            rows.append(("HemoSet test", arch, "joint (Fase 4)", h_joint))
        if r_base is not None:
            rows.append(("Rabbani test", arch, "single-source (Fase 1)", r_base))
        if r_joint is not None:
            rows.append(("Rabbani test", arch, "joint (Fase 4)", r_joint))

    if not rows:
        print("Nessun risultato trovato in", RESULTS_DIR)
        return

    header = "| Test set | Architettura | Training | " + " | ".join(m.upper() for m in METRICS) + " |"
    sep = "|---" * (3 + len(METRICS)) + "|"
    lines = [header, sep]
    for test_set, arch, training, metrics in rows:
        cells = " | ".join(fmt(metrics[m]) for m in METRICS)
        lines.append(f"| {test_set} | {arch} | {training} | {cells} |")

    table_md = "\n".join(lines)
    print(table_md)

    out_path = os.path.join(RESULTS_DIR, "phase4_joint_summary.md")
    with open(out_path, "w") as f:
        f.write("# Fase 4 - Joint training vs single-source (per dominio di test)\n\n")
        f.write("Media +/- deviazione standard sui 5 fold HemoSet; Rabbani single-source e' un run singolo.\n\n")
        f.write(table_md + "\n")

    print(f"\nSalvato anche in {out_path}")


if __name__ == "__main__":
    main()
