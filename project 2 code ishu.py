# REPO LINK -   https://github.com/ishu35060-max/HCL-.git
# Project 2: Student Academic Performance Analytics
# Dataset: 02_student_performance.csv

import os
import numpy as np
import pandas as pd

# --- Automatically download dataset if not present ---
if not os.path.exists("data"):
    os.makedirs("data")

# Clone/Download file from the repo if missing
if not os.path.exists("data/02_student_performance.csv"):
    print("Downloading dataset from Git repository...")
    os.system("git clone https://github.com/ishu35060-max/HCL-.git temp_git_repo")
    if os.path.exists("temp_git_repo/data/02_student_performance.csv"):
        os.rename("temp_git_repo/data/02_student_performance.csv", "data/02_student_performance.csv")
    elif os.path.exists("temp_git_repo/02_student_performance.csv"):
        os.rename("temp_git_repo/02_student_performance.csv", "data/02_student_performance.csv")
    os.system("rm -rf temp_git_repo")

# ---------------- CONFIG ----------------
FILE = "data/02_student_performance.csv"
ID_COL = "Student_ID"
DEPT_COL = "Department"
ATT_COL = "Attendance_Percent"
STUDY_COL = "Study_Hours_Per_Week"
NET_COL = "Internet_Access"
PYTHON_COL = "Python_Marks"
# ----------------------------------------

# 1. Load and inspect -----------------------------------------------
df = pd.read_csv(FILE)

print("Shape (rows, columns):", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary:\n", df.describe())

# every column that ends with _Marks is a subject sco
score_cols = [c for c in df.columns if c.endswith("_Marks")]
print("\nScore columns found:", score_cols)

# 2. Missing values and duplicate IDs -------------------------------
print("\nMissing values per column:\n", df.isnull().sum())

dup_ids = df[df.duplicated(subset=ID_COL, keep=False)]
print("\nRows with a repeated", ID_COL, ":", len(dup_ids))
if len(dup_ids) > 0:
    print(dup_ids.sort_values(ID_COL))
    # keep the first record of each student
    df = df.drop_duplicates(subset=ID_COL, keep="first").reset_index(drop=True)
    print("Shape after removing duplicate IDs:", df.shape)

# 3. Missing attendance ---------------------------------------------
# I fill with the median and not the mean, because the median is not pulled
# by a few students with very low or very high attendance.
missing_att = df[ATT_COL].isnull().sum()
att_median = df[ATT_COL].median()
df[ATT_COL] = df[ATT_COL].fillna(att_median)
print(f"\nFilled {missing_att} missing attendance values with the median ({att_median}).")

# 4. Scores outside 0-100 -------------------------------------------
print("\nChecking scores outside 0-100")
for col in score_cols:
    bad = df[(df[col] < 0) | (df[col] > 100)]
    print(f"  {col}: {len(bad)} invalid values")
    # treat invalid values as missing, then fill with the column median
    df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

for col in score_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# 5. Total and average marks ----------------------------------------
df["Total_Marks"] = df[score_cols].sum(axis=1)
df["Average_Marks"] = df[score_cols].mean(axis=1).round(2)
print("\nTotal and average marks (first 5):")
print(df[[ID_COL, "Total_Marks", "Average_Marks"]].head())

# 6. Grade column ---------------------------------------------------
# A: 90+, B: 75-89, C: 60-74, D: 40-59, F: below 40
bins = [0, 40, 60, 75, 90, 100]
labels = ["F", "D", "C", "B", "A"]
df["Grade"] = pd.cut(df["Average_Marks"], bins=bins, labels=labels,
                     right=False, include_lowest=True)
# right=False means 40 goes in D and 90 goes in A; 100 is covered below
df.loc[df["Average_Marks"] == 100, "Grade"] = "A"

print("\nGrade distribution:")
grade_count = df["Grade"].value_counts().reindex(labels[::-1])
print(grade_count)
print((grade_count / len(df) * 100).round(1).astype(str) + " %")

# 7. Department-wise average marks ----------------------------------
dept_summary = df.groupby(DEPT_COL).agg(
    Students=(ID_COL, "count"),
    Avg_Marks=("Average_Marks", "mean"),
    Avg_Attendance=(ATT_COL, "mean"),
    Avg_Study_Hours=(STUDY_COL, "mean"),
).round(2).sort_values("Avg_Marks", ascending=False)
print("\nDepartment summary:\n", dept_summary)

# 8. Top 10 students ------------------------------------------------
top10 = df.sort_values("Average_Marks", ascending=False).head(10)
print("\nTop 10 students by average marks:")
print(top10[[ID_COL, DEPT_COL, "Average_Marks", "Grade"]].to_string(index=False))

# 9. At-risk students (average below 40) ----------------------------
at_risk = df[df["Average_Marks" ] < 40]
print(f"\nStudents below 40% average: {len(at_risk)} "
      f"({len(at_risk) / len(df) * 100:.1f}% of all students)")
print(at_risk[[ID_COL, DEPT_COL, ATT_COL, STUDY_COL, "Average_Marks"]].to_string(index=False))

# 10. Attendance bands ----------------------------------------------
att_bins = [0, 60, 75, 90, 100]
att_labels = ["Below 60", "60-74", "75-89", "90-100"]
df["Attendance_Band"] = pd.cut(df[ATT_COL], bins=att_bins, labels=att_labels,
                               right=False, include_lowest=True)
df.loc[df[ATT_COL] == 100, "Attendance_Band"] = "90-100"

att_summary = df.groupby("Attendance_Band", observed=True)["Average_Marks"].agg(
    ["count", "mean"]).round(2)
print("\nAverage marks by attendance band:\n", att_summary)

# 11. Study hours vs marks ------------------------------------------
corr = df[STUDY_COL].corr(df["Average_Marks"])
print(f"\nCorrelation between study hours and average marks: {corr:.3f}")

# np.polyfit gives the slope of the best fit line
slope, intercept = np.polyfit(df[STUDY_COL], df["Average_Marks"], 1)
print(f"Best fit line: Average_Marks = {slope:.2f} * Study_Hours + {intercept:.2f}")

df["Study_Band"] = pd.cut(df[STUDY_COL], bins=[-0.01, 2, 4, 6, df[STUDY_COL].max() + 1],
                          labels=["0-2 hrs", "2-4 hrs", "4-6 hrs", "6+ hrs"])
print(df.groupby("Study_Band", observed=True)["Average_Marks"].agg(["count", "mean"]).round(2))

# 12. Internet access vs no internet --------------------------------
net_summary = df.groupby(NET_COL).agg(
    Students=(ID_COL, "count"),
    Avg_Marks=("Average_Marks", "mean"),
    Avg_Study_Hours=(STUDY_COL, "mean"),
    Avg_Attendance=(ATT_COL, "mean"),
).round(2)
print("\nInternet access comparison:\n", net_summary)

# 13. NumPy stats of Python_Marks -----------------------------------
py = df[PYTHON_COL].to_numpy()
q1, q2, q3 = np.percentile(py, [25, 50, 75])
print("\nPython_Marks (NumPy)")
print("  Mean              :", round(np.mean(py), 2))
print("  Median            :", np.median(py))
print("  Standard deviation:", round(np.std(py), 2))
print(f"  Q1 / Q2 / Q3      : {q1} / {q2} / {q3}")

# 14. Outliers with IQR ---------------------------------------------
iqr = q3 - q1
low_limit = q1 - 1.5 * iqr
high_limit = q3 + 1.5 * iqr
print(f"\nIQR = {iqr:.2f}, valid range = {low_limit:.2f} to {high_limit:.2f}")

py_outliers = df[(df[PYTHON_COL] < low_limit) | (df[PYTHON_COL] > high_limit)]
print("Python_Marks outliers:", len(py_outliers))
print(py_outliers[[ID_COL, PYTHON_COL]].to_string(index=False))

# same idea for the overall average marks
aq1, aq3 = np.percentile(df["Average_Marks"], [25, 75])
a_iqr = aq3 - aq1
avg_out = df[(df["Average_Marks"] < aq1 - 1.5 * a_iqr) |
             (df["Average_Marks"] > aq3 + 1.5 * a_iqr)]
print("Average_Marks outliers:", len(avg_out))

# 15. Recommendations -----------------------------------------------
# These are generated from the numbers, so they change with the data.
best_dept = dept_summary.index[0]
worst_dept = dept_summary.index[-1]
low_att = att_summary["mean"].iloc[0]
high_att = att_summary["mean"].iloc[-1]

print("\n========== RECOMMENDATIONS ==========")
print(f"1. {len(at_risk)} students have an average below 40%. Give them mentoring "
      "and extra practice sessions early in the semester.")
print(f"2. {worst_dept} has the lowest average marks ({dept_summary['Avg_Marks'].iloc[-1]}) "
      f"while {best_dept} has the highest ({dept_summary['Avg_Marks'].iloc[0]}). "
      f"Share teaching methods from {best_dept} and support {worst_dept}.")
print(f"3. Students in the lowest attendance band average {low_att}, compared with "
      f"{high_att} in the highest band. Track attendance weekly and contact "
      "students who keep dropping.")
print(f"4. Study hours and marks have a correlation of {corr:.2f}. "
      "Encourage regular study schedules, but this alone does not prove "
      "that more study causes better marks.")
print("5. Compare internet and non-internet groups above. If the gap is large, "
      "offer lab hours or offline study material for students without internet.")
print("Note: this is a synthetic dataset, so these are patterns, not proof of cause.")

# Save cleaned data and the main tables
df.to_csv("data/cleaned_student_performance.csv", index=False)
dept_summary.to_csv("data/department_summary.csv")
at_risk.to_csv("data/at_risk_students.csv", index=False)
print("\nSaved cleaned data, department summary and at-risk list.")