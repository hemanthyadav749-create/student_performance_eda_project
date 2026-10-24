"""
Student Performance Analysis using Python and Exploratory Data Analysis (EDA)
Dataset: UCI Student Performance (student-mat.csv)
Run: python src/analyze_students.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student-mat.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data(path: Path) -> pd.DataFrame:
    """Load the UCI CSV, which uses semicolons as separators."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the UCI Student Performance dataset and copy student-mat.csv "
            "into the project's data/ folder. Dataset page: "
            "https://archive.ics.uci.edu/dataset/320/student+performance"
        )
    return pd.read_csv(path, sep=";")


def main():
    df = load_data(DATA_PATH)

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nNumeric summary:")
    print(df.describe().round(2).to_string())

    # Clean exact duplicate rows, if any.
    df = df.drop_duplicates().copy()

    # 1. Distribution of final grades
    plt.figure(figsize=(8, 5))
    sns.histplot(df["G3"], bins=11, kde=True)
    plt.title("Distribution of Final Mathematics Grades (G3)")
    plt.xlabel("Final grade (0–20)")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_final_grade_distribution.png", dpi=150)
    plt.close()

    # 2. Average final grade by weekly study-time category
    study_means = df.groupby("studytime", as_index=False)["G3"].mean()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=study_means, x="studytime", y="G3")
    plt.title("Average Final Grade by Study-Time Category")
    plt.xlabel("Study time (1 = lowest, 4 = highest)")
    plt.ylabel("Average final grade")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_studytime_vs_grade.png", dpi=150)
    plt.close()

    # 3. Absences vs final grade
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="absences", y="G3", alpha=0.7)
    plt.title("Absences vs Final Mathematics Grade")
    plt.xlabel("Number of absences")
    plt.ylabel("Final grade (0–20)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_absences_vs_grade.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap for selected numeric columns
    numeric_cols = ["age", "studytime", "failures", "absences", "G1", "G2", "G3"]
    plt.figure(figsize=(9, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Selected Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Compare average grade by past failures
    failure_means = df.groupby("failures", as_index=False)["G3"].mean()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=failure_means, x="failures", y="G3")
    plt.title("Average Final Grade by Number of Past Failures")
    plt.xlabel("Past class failures")
    plt.ylabel("Average final grade")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_failures_vs_grade.png", dpi=150)
    plt.close()

    # Save summary tables for use in the report.
    study_means.to_csv(OUTPUT_DIR / "average_grade_by_studytime.csv", index=False)
    failure_means.to_csv(OUTPUT_DIR / "average_grade_by_failures.csv", index=False)

    print("\n=== KEY SUMMARY VALUES ===")
    print(f"Average final grade (G3): {df['G3'].mean():.2f}/20")
    print(f"Median final grade (G3): {df['G3'].median():.2f}/20")
    print(f"Students with G3 = 0: {(df['G3'] == 0).sum()}")
    print("\nAverage final grade by study time:")
    print(study_means.round(2).to_string(index=False))
    print("\nAverage final grade by past failures:")
    print(failure_means.round(2).to_string(index=False))
    print(f"\nDone! Charts and summary tables are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
