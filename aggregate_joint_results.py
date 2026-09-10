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


def joint_metrics(arch, sampling="natural"):
    suffix = "" if sampling == "natural" else f"_{sampling}"
    runs = [load(f"joint_{arch}_fold{i}{suffix}") for i in range(5)]
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
        h_joint, r_joint = joint_metrics(arch, "natural")
        h_joint_bal, r_joint_bal = joint_metrics(arch, "balanced")

        if h_base is not None:
            rows.append(("HemoSet test", arch, "single-source (Phase 1)", h_base))
        if h_joint is not None:
            rows.append(("HemoSet test", arch, "natural joint (Phase 4)", h_joint))
        if h_joint_bal is not None:
            rows.append(("HemoSet test", arch, "balanced joint (Phase 4b)", h_joint_bal))
        if r_base is not None:
            rows.append(("Rabbani test", arch, "single-source (Phase 1)", r_base))
        if r_joint is not None:
            rows.append(("Rabbani test", arch, "natural joint (Phase 4)", r_joint))
        if r_joint_bal is not None:
            rows.append(("Rabbani test", arch, "balanced joint (Phase 4b)", r_joint_bal))

    if not rows:
        print("No results found in", RESULTS_DIR)
        return

    header = "| Test set | Architecture | Training | " + " | ".join(m.upper() for m in METRICS) + " |"
    sep = "|---" * (3 + len(METRICS)) + "|"
    lines = [header, sep]
    for test_set, arch, training, metrics in rows:
        cells = " | ".join(fmt(metrics[m]) for m in METRICS)
        lines.append(f"| {test_set} | {arch} | {training} | {cells} |")

    table_md = "\n".join(lines)
    print(table_md)

    out_path = os.path.join(RESULTS_DIR, "phase4_joint_summary.md")
    with open(out_path, "w") as f:
        f.write("# Phase 4/4b - Joint training (natural vs balanced) vs single-source\n\n")
        f.write("Mean +/- standard deviation over the 5 HemoSet folds; Rabbani single-source is a single run.\n\n")
        f.write(table_md + "\n")

    print(f"\nAlso saved to {out_path}")


if __name__ == "__main__":
    main()
