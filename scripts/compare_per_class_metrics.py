from pathlib import Path
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare per-class metrics between two experiments."
    )

    parser.add_argument(
        "csv_a",
        type=Path,
        help="Path to the first per-class metrics CSV.",
    )

    parser.add_argument(
        "csv_b",
        type=Path,
        help="Path to the second per-class metrics CSV.",
    )

    parser.add_argument(
        "--name-a",
        default="exp_a",
        help="Name for the first experiment.",
    )

    parser.add_argument(
        "--name-b",
        default="exp_b",
        help="Name for the second experiment.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the comparison CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)

    comparison = df_a.merge(
        df_b,
        on="class_name",
        suffixes=(f"_{args.name_a}", f"_{args.name_b}"),
    )

    comparison["recall_diff"] = (
        comparison[f"recall_{args.name_b}"]
        - comparison[f"recall_{args.name_a}"]
    )

    comparison["f1_diff"] = (
        comparison[f"f1_score_{args.name_b}"]
        - comparison[f"f1_score_{args.name_a}"]
    )

    comparison = comparison[
        [
            "class_name",
            f"support_{args.name_a}",
            f"recall_{args.name_a}",
            f"recall_{args.name_b}",
            "recall_diff",
            f"f1_score_{args.name_a}",
            f"f1_score_{args.name_b}",
            "f1_diff",
        ]
    ]

    if args.output is None:
        output_path = Path(
            f"{args.name_a}_vs_{args.name_b}_per_class_comparison.csv"
        )
    else:
        output_path = args.output

    comparison.to_csv(
        output_path,
        index=False,
    )

    print(comparison.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()