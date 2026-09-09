from pathlib import Path

import argparse

import pandas as pd

import matplotlib.pyplot as plt

def parse_args():

    parser = argparse.ArgumentParser(

        description="Plot per-class metric comparison between two experiments."

    )

    parser.add_argument(

        "comparison_csv",

        type=Path,

        help="Path to the per-class comparison CSV.",

    )

    parser.add_argument(

        "--name-a",

        default="exp_a",

        help="Label for the first experiment.",

    )

    parser.add_argument(

        "--name-b",

        default="exp_b",

        help="Label for the second experiment.",

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=Path("outputs/figures"),

        help="Directory to save figures.",

    )

    return parser.parse_args()

def plot_metric(df, metric, name_a, name_b, output_path):

    classes = df["class_name"]

    values_a = df[f"{metric}_{name_a}"]

    values_b = df[f"{metric}_{name_b}"]

    y = range(len(classes))

    bar_height = 0.4

    plt.figure(figsize=(10, 8))

    plt.barh(

        [i - bar_height / 2 for i in y],

        values_a,

        height=bar_height,

        label=name_a,

    )

    plt.barh(

        [i + bar_height / 2 for i in y],

        values_b,

        height=bar_height,

        label=name_b,

    )

    plt.yticks(y, classes)

    plt.xlabel(metric.capitalize())

    plt.ylabel("Tissue class")

    plt.title(

        f"Per-class {metric.capitalize()}: "

        f"{name_a} vs {name_b}"

    )

    plt.xlim(0, 1.05)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

def main():

    args = parse_args()

    df = pd.read_csv(args.comparison_csv)

    args.output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    recall_path = (

        args.output_dir

        / f"{args.name_a}_vs_{args.name_b}_recall.png"

    )

    f1_path = (

        args.output_dir

        / f"{args.name_a}_vs_{args.name_b}_f1.png"

    )

    plot_metric(

        df,

        "recall",

        args.name_a,

        args.name_b,

        recall_path,

    )

    plot_metric(

        df,

        "f1_score",

        args.name_a,

        args.name_b,

        f1_path,

    )

    print(f"Saved: {recall_path}")

    print(f"Saved: {f1_path}")

if __name__ == "__main__":

    main()