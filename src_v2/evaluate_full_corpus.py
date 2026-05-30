import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from .config import (
        ARTIFACT_DIR,
        FIELDS,
        GROUND_TRUTH_PATH,
        IMAGE_DIR,
        TARGET_LATENCY_SECONDS,
    )
    from .evaluate_v2 import is_match, load_rows
    from .inference_v2 import ReceiptFieldExtractorV2
except ImportError:
    from config import (
        ARTIFACT_DIR,
        FIELDS,
        GROUND_TRUTH_PATH,
        IMAGE_DIR,
        TARGET_LATENCY_SECONDS,
    )
    from evaluate_v2 import is_match, load_rows
    from inference_v2 import ReceiptFieldExtractorV2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(math.ceil(q * len(sorted_values))) - 1
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _image_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name.lower())
    return (1, stem.lower(), path.name.lower())


def _load_font(size: int):
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_vertical_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    y_label: str,
    output_path: Path,
    value_fmt: str = "{:.2f}",
    y_max: Optional[float] = None,
    target_line: Optional[float] = None,
    bar_color: Tuple[int, int, int] = (41, 98, 255),
):
    if not labels:
        return

    y_cap = y_max if y_max is not None else max(values + [0.0])
    if y_cap <= 0:
        y_cap = 1.0
    if target_line is not None:
        y_cap = max(y_cap, target_line)
    y_cap *= 1.15

    width = max(980, 120 + len(labels) * 130)
    height = 680
    margin_left = 100
    margin_right = 40
    margin_top = 90
    margin_bottom = 170

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    axis_font = _load_font(18)
    label_font = _load_font(16)
    value_font = _load_font(16)

    draw.text((margin_left, 30), title, fill=(20, 20, 20), font=title_font)

    # Axes
    x0 = margin_left
    y0 = margin_top + plot_h
    draw.line([(x0, margin_top), (x0, y0), (x0 + plot_w, y0)], fill=(90, 90, 90), width=2)

    # Grid + ticks
    for i in range(6):
        frac = i / 5
        y = y0 - int(frac * plot_h)
        val = frac * y_cap
        draw.line([(x0, y), (x0 + plot_w, y)], fill=(232, 232, 232), width=1)
        draw.text((16, y - 8), value_fmt.format(val), fill=(90, 90, 90), font=label_font)

    if target_line is not None:
        y_target = y0 - int((target_line / y_cap) * plot_h)
        draw.line([(x0, y_target), (x0 + plot_w, y_target)], fill=(220, 53, 69), width=2)
        draw.text((x0 + 8, y_target - 24), f"Target: {value_fmt.format(target_line)}", fill=(180, 40, 52), font=label_font)

    slot_w = plot_w / max(len(labels), 1)
    bar_w = int(slot_w * 0.62)

    for idx, (label, value) in enumerate(zip(labels, values)):
        x_center = x0 + int((idx + 0.5) * slot_w)
        x_left = x_center - bar_w // 2
        x_right = x_left + bar_w
        bar_h = int((max(value, 0.0) / y_cap) * plot_h)
        y_top = y0 - bar_h

        draw.rectangle([(x_left, y_top), (x_right, y0)], fill=bar_color, outline=(28, 73, 206))
        draw.text((x_left, y_top - 22), value_fmt.format(value), fill=(40, 40, 40), font=value_font)

        # Keep label readable by wrapping one time if needed.
        short_label = label
        if len(short_label) > 14 and "_" in short_label:
            parts = short_label.split("_")
            short_label = "\n".join(["_".join(parts[:2]), "_".join(parts[2:])]) if len(parts) > 2 else short_label
        elif len(short_label) > 16:
            short_label = short_label[:15] + "…"
        draw.multiline_text((x_center - 55, y0 + 12), short_label, fill=(70, 70, 70), font=label_font, align="center")

    draw.text((20, margin_top + plot_h // 2), y_label, fill=(70, 70, 70), font=axis_font)
    img.save(output_path)


def _draw_horizontal_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    x_label: str,
    output_path: Path,
    value_fmt: str = "{:.0f}",
    bar_color: Tuple[int, int, int] = (0, 150, 136),
):
    if not labels:
        return

    width = 1320
    row_h = 34
    height = max(520, 140 + len(labels) * row_h)
    margin_left = 360
    margin_right = 70
    margin_top = 90
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_value = max(values + [1.0]) * 1.12

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    label_font = _load_font(16)
    value_font = _load_font(15)

    draw.text((40, 30), title, fill=(20, 20, 20), font=title_font)

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * row_h
        y_mid = y + row_h // 2
        bar_len = int((value / max_value) * plot_w)
        draw.rectangle([(margin_left, y + 7), (margin_left + bar_len, y + row_h - 7)], fill=bar_color, outline=(0, 120, 110))

        short_label = label if len(label) <= 42 else label[:41] + "…"
        draw.text((20, y + 8), short_label, fill=(70, 70, 70), font=label_font)
        draw.text((margin_left + bar_len + 10, y + 8), value_fmt.format(value), fill=(40, 40, 40), font=value_font)

        if idx % 2 == 0:
            draw.line([(margin_left, y_mid), (margin_left + plot_w, y_mid)], fill=(244, 244, 244), width=1)

    draw.text((margin_left, height - 40), x_label, fill=(70, 70, 70), font=label_font)
    img.save(output_path)


def _draw_latency_histogram(
    latencies: List[float],
    output_path: Path,
    bins: int = 20,
):
    if not latencies:
        return

    values = np.array(latencies, dtype=float)
    counts, edges = np.histogram(values, bins=bins)

    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(counts))]
    heights = counts.astype(float).tolist()

    _draw_vertical_bar_chart(
        labels=labels,
        values=heights,
        title="Latency Distribution (All Evaluated Images)",
        y_label="Image Count",
        output_path=output_path,
        value_fmt="{:.0f}",
    )


def _build_visualizations(summary: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Supervised accuracy per field (labeled subset)
    supervised = summary.get("supervised_accuracy", {})
    field_stats = supervised.get("per_field", {})
    labels = list(field_stats.keys())
    acc_values = [float(field_stats[f].get("accuracy", 0.0)) for f in labels]
    _draw_vertical_bar_chart(
        labels=labels,
        values=acc_values,
        title=f"Field Accuracy on Labeled Subset ({supervised.get('labeled_images', 0)} images)",
        y_label="Accuracy",
        output_path=output_dir / "chart_supervised_field_accuracy_full.png",
        value_fmt="{:.2f}",
        y_max=1.0,
    )

    # 2) Operational fill-rate across all images
    operational = summary.get("operational", {}).get("per_field", {})
    op_labels = list(operational.keys())
    fill_values = [float(operational[f].get("fill_rate", 0.0)) for f in op_labels]
    _draw_vertical_bar_chart(
        labels=op_labels,
        values=fill_values,
        title="Field Fill Rate on Full Corpus (7008 images)",
        y_label="Fill Rate",
        output_path=output_dir / "chart_field_fill_rate_full.png",
        value_fmt="{:.2f}",
        y_max=1.0,
    )

    # 3) Needs-review rate across all images
    review_values = [float(operational[f].get("needs_review_rate", 0.0)) for f in op_labels]
    _draw_vertical_bar_chart(
        labels=op_labels,
        values=review_values,
        title="Needs-Review Rate on Full Corpus (7008 images)",
        y_label="Needs Review Rate",
        output_path=output_dir / "chart_field_review_rate_full.png",
        value_fmt="{:.2f}",
        y_max=max(max(review_values + [0.05]), 0.2),
    )

    # 4) Latency histogram on all images
    _draw_latency_histogram(
        latencies=summary.get("latency", {}).get("all_values", []),
        output_path=output_dir / "chart_latency_histogram_full.png",
    )

    # 5) Template distribution top 20
    template_items = summary.get("template_distribution", {}).get("top_templates", [])
    if template_items:
        top_templates = template_items[:20]
        _draw_horizontal_bar_chart(
            labels=[item["template"] for item in top_templates],
            values=[float(item["count"]) for item in top_templates],
            title="Template Distribution (Top 20) - Full Corpus",
            x_label="Image Count",
            output_path=output_dir / "chart_template_distribution_top20_full.png",
            value_fmt="{:.0f}",
        )


def _summarize_and_save(
    rows_df: pd.DataFrame,
    output_dir: Path,
    started_at: float,
    image_count_total: int,
    supervised_field_stats: Dict[str, Dict[str, int]],
    labeled_images_evaluated: int,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    latencies = [float(x) for x in rows_df["latency_seconds"].dropna().tolist()]
    lat_sorted = sorted(latencies)

    if latencies:
        latency_stats = {
            "mean": round(statistics.mean(latencies), 4),
            "median": round(statistics.median(latencies), 4),
            "p90": round(_percentile(lat_sorted, 0.90), 4),
            "p95": round(_percentile(lat_sorted, 0.95), 4),
            "p99": round(_percentile(lat_sorted, 0.99), 4),
            "max": round(max(latencies), 4),
            "under_1s_ratio": round(sum(1 for x in latencies if x < TARGET_LATENCY_SECONDS) / len(latencies), 4),
            "all_values": latencies,
        }
    else:
        latency_stats = {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "under_1s_ratio": 0.0,
            "all_values": [],
        }

    operational_per_field = {}
    for field in FIELDS:
        pred_col = f"pred_{field}"
        conf_col = f"conf_{field}"
        review_col = f"review_{field}"

        non_null_mask = rows_df[pred_col].notna() & (rows_df[pred_col].astype(str) != "")
        fill_count = int(non_null_mask.sum())
        total_images = int(len(rows_df))
        fill_rate = (fill_count / total_images) if total_images else 0.0

        conf_values = pd.to_numeric(rows_df.loc[non_null_mask, conf_col], errors="coerce").dropna().tolist()
        review_values = pd.to_numeric(rows_df[review_col], errors="coerce").fillna(0.0)
        review_rate = float(review_values.mean()) if len(review_values) else 0.0

        operational_per_field[field] = {
            "fill_count": fill_count,
            "fill_rate": round(fill_rate, 4),
            "avg_confidence_on_filled": round(float(np.mean(conf_values)), 4) if conf_values else 0.0,
            "p10_confidence_on_filled": round(float(np.percentile(conf_values, 10)), 4) if conf_values else 0.0,
            "p50_confidence_on_filled": round(float(np.percentile(conf_values, 50)), 4) if conf_values else 0.0,
            "p90_confidence_on_filled": round(float(np.percentile(conf_values, 90)), 4) if conf_values else 0.0,
            "needs_review_rate": round(review_rate, 4),
        }

    per_field_sup = {}
    total_correct = 0
    total_count = 0
    total_miss = 0
    total_wrong = 0
    total_over_extract = 0

    for field in FIELDS:
        stat = dict(supervised_field_stats.get(field, {}))
        stat.setdefault("total", 0)
        stat.setdefault("correct", 0)
        stat.setdefault("miss", 0)
        stat.setdefault("wrong_value", 0)
        stat.setdefault("over_extract_on_gt_null", 0)
        stat["accuracy"] = round((stat["correct"] / stat["total"]), 4) if stat["total"] else 0.0
        per_field_sup[field] = stat

        total_correct += int(stat["correct"])
        total_count += int(stat["total"])
        total_miss += int(stat["miss"])
        total_wrong += int(stat["wrong_value"])
        total_over_extract += int(stat["over_extract_on_gt_null"])

    template_counter = Counter(rows_df["template"].fillna("unknown").replace("", "unknown").tolist())
    top_templates = [{"template": k, "count": int(v)} for k, v in template_counter.most_common(30)]

    duration_seconds = round(time.perf_counter() - started_at, 4)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "generated_at": now,
        "dataset": {
            "images_total_in_folder": int(image_count_total),
            "images_evaluated": int(len(rows_df)),
            "images_with_ground_truth": int(rows_df["is_labeled"].sum()),
            "labeled_images_evaluated": int(labeled_images_evaluated),
            "unlabeled_images_evaluated": int(len(rows_df) - labeled_images_evaluated),
        },
        "runtime": {
            "duration_seconds": duration_seconds,
            "duration_minutes": round(duration_seconds / 60.0, 2),
        },
        "latency": latency_stats,
        "operational": {
            "per_field": operational_per_field,
            "error_images_count": int((rows_df["inference_error"] == 1).sum()),
        },
        "supervised_accuracy": {
            "labeled_images": int(labeled_images_evaluated),
            "total_comparisons": int(total_count),
            "overall_accuracy": round((total_correct / total_count), 4) if total_count else 0.0,
            "total_correct": int(total_correct),
            "total_miss": int(total_miss),
            "total_wrong_value": int(total_wrong),
            "total_over_extract_on_gt_null": int(total_over_extract),
            "per_field": per_field_sup,
        },
        "template_distribution": {
            "unique_templates": int(len(template_counter)),
            "top_templates": top_templates,
        },
    }

    # Save summary with and without raw latency array
    summary_with_raw = json.loads(json.dumps(summary))
    with (output_dir / "summary_full_7008.json").open("w", encoding="utf-8") as fp:
        json.dump(summary_with_raw, fp, ensure_ascii=False, indent=2)

    summary_without_raw = json.loads(json.dumps(summary))
    summary_without_raw["latency"].pop("all_values", None)
    with (output_dir / "summary_full_7008_compact.json").open("w", encoding="utf-8") as fp:
        json.dump(summary_without_raw, fp, ensure_ascii=False, indent=2)

    # Generate human-readable markdown insight
    md_lines = [
        "# Full Corpus Evaluation Report (7008 Images)",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Images evaluated: `{summary['dataset']['images_evaluated']}` / `{summary['dataset']['images_total_in_folder']}`",
        f"- Labeled images: `{summary['dataset']['labeled_images_evaluated']}`",
        f"- Unlabeled images: `{summary['dataset']['unlabeled_images_evaluated']}`",
        "",
        "## Supervised Accuracy (Labeled Subset)",
        f"- Overall accuracy: **{summary['supervised_accuracy']['overall_accuracy']:.2%}**",
        f"- Total comparisons: `{summary['supervised_accuracy']['total_comparisons']}`",
        f"- Miss: `{summary['supervised_accuracy']['total_miss']}`, Wrong: `{summary['supervised_accuracy']['total_wrong_value']}`, Over-extract on GT null: `{summary['supervised_accuracy']['total_over_extract_on_gt_null']}`",
        "",
        "## Latency (All Evaluated Images)",
        f"- Mean: `{summary['latency']['mean']:.4f}s`",
        f"- Median: `{summary['latency']['median']:.4f}s`",
        f"- P90/P95/P99: `{summary['latency']['p90']:.4f}s / {summary['latency']['p95']:.4f}s / {summary['latency']['p99']:.4f}s`",
        f"- Max: `{summary['latency']['max']:.4f}s`",
        f"- Ratio under `{TARGET_LATENCY_SECONDS:.1f}s`: `{summary['latency']['under_1s_ratio']:.2%}`",
        "",
        "## Operational Field Health (All Images)",
    ]

    for field in FIELDS:
        stat = summary["operational"]["per_field"][field]
        md_lines.append(
            f"- `{field}`: fill_rate={stat['fill_rate']:.2%}, "
            f"avg_conf={stat['avg_confidence_on_filled']:.4f}, "
            f"needs_review_rate={stat['needs_review_rate']:.2%}"
        )

    md_lines.extend(
        [
            "",
            "## Top Template Distribution",
        ]
    )
    for item in summary["template_distribution"]["top_templates"][:15]:
        md_lines.append(f"- `{item['template']}`: {item['count']} images")

    with (output_dir / "report_full_7008.md").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines) + "\n")

    _build_visualizations(summary_with_raw, output_dir=output_dir)
    return summary_without_raw


def evaluate_full_corpus(
    image_dir: Path,
    ground_truth_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    progress_every: int = 100,
) -> Dict:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images = sorted(images, key=_image_sort_key)
    if limit is not None:
        images = images[:limit]

    gt_rows = load_rows(ground_truth_path)
    gt_map = {str(row.get("image", "")): row.get("ground_truth", {}) for row in gt_rows}

    extractor = ReceiptFieldExtractorV2()
    all_records: List[Dict] = []
    labeled_images_evaluated = 0
    supervised_field_stats = {
        field: {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "wrong_value": 0,
            "over_extract_on_gt_null": 0,
        }
        for field in FIELDS
    }

    print(f"[INFO] Evaluating images: {len(images)}")
    print(f"[INFO] Ground-truth rows available: {len(gt_map)}")
    print(f"[INFO] Output dir: {output_dir}")

    for idx, image_path in enumerate(images, start=1):
        image_name = image_path.name
        gt = gt_map.get(image_name)
        is_labeled = 1 if gt is not None else 0
        if is_labeled:
            labeled_images_evaluated += 1

        record = {
            "image": image_name,
            "is_labeled": is_labeled,
            "template": None,
            "template_score": None,
            "latency_seconds": None,
            "inference_error": 0,
            "error_message": "",
        }

        for field in FIELDS:
            record[f"pred_{field}"] = None
            record[f"conf_{field}"] = None
            record[f"review_{field}"] = None
            record[f"source_{field}"] = None
            record[f"gt_{field}"] = gt.get(field) if gt is not None else None

        try:
            meta = extractor.predict(str(image_path), return_meta=True)
            pred = meta.get("data", {})
            conf = meta.get("confidence", {})
            review = meta.get("needs_review", {})
            source = meta.get("source", {})

            record["template"] = meta.get("template") or "unknown"
            record["template_score"] = _safe_float(meta.get("template_score"), 0.0)
            record["latency_seconds"] = _safe_float(meta.get("latency_seconds"), 0.0)

            for field in FIELDS:
                record[f"pred_{field}"] = pred.get(field)
                record[f"conf_{field}"] = _safe_float(conf.get(field), np.nan)
                record[f"review_{field}"] = 1 if bool(review.get(field)) else 0
                record[f"source_{field}"] = source.get(field)

            if gt is not None:
                for field in FIELDS:
                    gt_value = gt.get(field)
                    pred_value = pred.get(field)
                    if gt_value is None:
                        if pred_value is not None:
                            supervised_field_stats[field]["over_extract_on_gt_null"] += 1
                        continue

                    supervised_field_stats[field]["total"] += 1
                    if is_match(field, pred_value, gt_value):
                        supervised_field_stats[field]["correct"] += 1
                    elif pred_value is None:
                        supervised_field_stats[field]["miss"] += 1
                    else:
                        supervised_field_stats[field]["wrong_value"] += 1
        except Exception as exc:
            record["inference_error"] = 1
            record["error_message"] = str(exc)

        all_records.append(record)

        if (idx % progress_every == 0) or (idx == len(images)):
            elapsed = time.perf_counter() - started_at
            speed = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(images) - idx) / speed if speed > 0 else 0.0
            print(
                f"[PROGRESS] {idx}/{len(images)} "
                f"({(idx/len(images))*100:.2f}%) | "
                f"elapsed={elapsed/60:.2f}m | "
                f"avg={speed:.2f} img/s | "
                f"eta={eta/60:.2f}m"
            )

    rows_df = pd.DataFrame(all_records)
    row_csv_path = output_dir / "row_level_full_7008.csv"
    rows_df.to_csv(row_csv_path, index=False)
    print(f"[INFO] Row-level saved: {row_csv_path}")

    field_csv_path = output_dir / "field_level_labeled_full_7008.csv"
    field_rows = []
    for field in FIELDS:
        stat = supervised_field_stats[field]
        total = int(stat["total"])
        correct = int(stat["correct"])
        miss = int(stat["miss"])
        wrong = int(stat["wrong_value"])
        over_extract = int(stat["over_extract_on_gt_null"])

        field_rows.append(
            {
                "field": field,
                "total": total,
                "correct": correct,
                "accuracy": (correct / total) if total else 0.0,
                "miss": miss,
                "wrong_value": wrong,
                "over_extract_on_gt_null": over_extract,
            }
        )
    pd.DataFrame(field_rows).to_csv(field_csv_path, index=False)
    print(f"[INFO] Field-level saved: {field_csv_path}")

    summary = _summarize_and_save(
        rows_df=rows_df,
        output_dir=output_dir,
        started_at=started_at,
        image_count_total=len(images),
        supervised_field_stats=supervised_field_stats,
        labeled_images_evaluated=labeled_images_evaluated,
    )
    print(f"[INFO] Summary saved: {output_dir / 'summary_full_7008_compact.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on full image corpus and generate visual report.")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR), help="Path to image directory")
    parser.add_argument("--ground-truth", type=str, default=str(GROUND_TRUTH_PATH), help="Path to ground-truth jsonl")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ARTIFACT_DIR / "evaluation_visuals_full_7008"),
        help="Output directory for csv/json/png report artifacts",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debug run")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N images")
    args = parser.parse_args()

    summary = evaluate_full_corpus(
        image_dir=Path(args.image_dir),
        ground_truth_path=Path(args.ground_truth),
        output_dir=Path(args.out_dir),
        limit=args.limit,
        progress_every=args.progress_every,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
