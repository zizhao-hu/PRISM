"""Collect and summarize MMLU + MT-Bench baseline results."""

import argparse
import json
import os
import glob


# MMLU category groupings
MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history", "international_law",
        "jurisprudence", "logical_fallacies", "moral_disputes",
        "moral_scenarios", "philosophy", "prehistory", "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}


def collect_mmlu(result_root):
    """Collect MMLU results per model and per category."""
    mmlu_dir = os.path.join(result_root, "mmlu")
    if not os.path.exists(mmlu_dir):
        print("No MMLU results found")
        return {}

    all_results = {}
    for model_dir in sorted(glob.glob(os.path.join(mmlu_dir, "*"))):
        model_name = os.path.basename(model_dir)

        # lm-eval-harness saves results in a specific structure
        result_files = glob.glob(os.path.join(model_dir, "**", "results.json"), recursive=True)
        if not result_files:
            print(f"  No results for {model_name}")
            continue

        d = json.load(open(result_files[0]))
        results = d.get("results", {})

        # Extract per-subject scores
        subject_scores = {}
        for key, val in results.items():
            # Keys like "mmlu_abstract_algebra" or "hendrycksTest-abstract_algebra"
            subject = key.replace("mmlu_", "").replace("hendrycksTest-", "")
            if isinstance(val, dict):
                acc = val.get("acc,none", val.get("acc", val.get("acc_norm,none", 0)))
                if acc:
                    subject_scores[subject] = acc

        # Group by category
        category_scores = {}
        for cat, subjects in MMLU_CATEGORIES.items():
            scores = [subject_scores[s] for s in subjects if s in subject_scores]
            if scores:
                category_scores[cat] = sum(scores) / len(scores)

        overall = sum(subject_scores.values()) / len(subject_scores) if subject_scores else 0

        all_results[model_name] = {
            "overall": round(overall * 100, 1),
            "per_category": {k: round(v * 100, 1) for k, v in category_scores.items()},
            "per_subject": {k: round(v * 100, 1) for k, v in subject_scores.items()},
            "n_subjects": len(subject_scores),
        }
        print(f"\n{model_name}: MMLU Overall = {overall*100:.1f}%")
        for cat, score in sorted(category_scores.items()):
            print(f"  {cat}: {score*100:.1f}%")

    return all_results


def collect_mt_bench(result_root):
    """Collect MT-Bench results per model and per category."""
    mt_dir = os.path.join(result_root, "mt_bench")
    if not os.path.exists(mt_dir):
        print("No MT-Bench results found")
        return {}

    all_results = {}
    for model_dir in sorted(glob.glob(os.path.join(mt_dir, "*"))):
        model_name = os.path.basename(model_dir)
        summary_file = os.path.join(model_dir, "mt_bench_summary.json")
        if not os.path.exists(summary_file):
            print(f"  No MT-Bench summary for {model_name}")
            continue

        d = json.load(open(summary_file))
        all_results[model_name] = d
        print(f"\n{model_name}: MT-Bench Overall = {d.get('overall', 0):.2f}")
        for cat, score in sorted(d.get("per_category", {}).items()):
            print(f"  {cat}: {score:.2f}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="results/baselines_7b8b")
    args = parser.parse_args()

    print("=" * 60)
    print("MMLU Results")
    print("=" * 60)
    mmlu = collect_mmlu(args.result_root)

    print("\n" + "=" * 60)
    print("MT-Bench Results")
    print("=" * 60)
    mt_bench = collect_mt_bench(args.result_root)

    # Save combined summary
    summary = {"mmlu": mmlu, "mt_bench": mt_bench}
    out_path = os.path.join(args.result_root, "combined_summary.json")
    os.makedirs(args.result_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined summary to {out_path}")


if __name__ == "__main__":
    main()
