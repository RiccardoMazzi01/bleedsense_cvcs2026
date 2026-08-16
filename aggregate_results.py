import json
import os
import statistics as stats

RESULTS_DIR = "/work/cvcs2026/bleedsense/results"
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95"]


def load(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


def mean_std(values):
    m = stats.mean(values)
    s = stats.stdev(values) if len(values) > 1 else 0.0
    return m, s


def fmt(values):
    m, s = mean_std(values)
    if len(values) > 1:
        return f"{m:.3f}±{s:.3f}"
    return f"{m:.3f}"


def summarize_hemoset(arch):
    runs = [load(f"hemoset_{arch}_fold{i}") for i in range(5)]
    indomain = {m: [r["indomain_test"][m] for r in runs] for m in METRICS}
    cross = {m: [r["cross_dataset_test"][m] for r in runs] for m in METRICS}
    return indomain, cross


def summarize_rabbani(arch):
    run = load(f"rabbani_{arch}")
    indomain = {m: [run["indomain_test"][m]] for m in METRICS}
    cross = {m: [run["cross_dataset_test"][m]] for m in METRICS}
    return indomain, cross


def main():
    rows = []
    for arch in ARCHITECTURES:
        h_in, h_cross = summarize_hemoset(arch)
        r_in, r_cross = summarize_rabbani(arch)
        rows.append(("HemoSet -> HemoSet (in-domain)", arch, h_in))
        rows.append(("HemoSet -> Rabbani (cross)", arch, h_cross))
        rows.append(("Rabbani -> Rabbani (in-domain)", arch, r_in))
        rows.append(("Rabbani -> HemoSet (cross)", arch, r_cross))

    header = "| Setting | Architettura | " + " | ".join(m.upper() for m in METRICS) + " |"
    sep = "|---" * (2 + len(METRICS)) + "|"
    lines = [header, sep]
    for setting, arch, metrics in rows:
        cells = " | ".join(fmt(metrics[m]) for m in METRICS)
        lines.append(f"| {setting} | {arch} | {cells} |")

    table_md = "\n".join(lines)
    print(table_md)

    out_path = os.path.join(RESULTS_DIR, "phase1_summary.md")
    with open(out_path, "w") as f:
        f.write("# Fase 1 - Risultati baseline (in-domain vs cross-dataset)\n\n")
        f.write("Media +/- deviazione standard sui 5 fold per HemoSet; run singolo per Rabbani.\n\n")
        f.write(table_md + "\n")

    print(f"\nSalvato anche in {out_path}")


if __name__ == "__main__":
    main()
