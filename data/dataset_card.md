# Dataset Card: Software Release Defect History (RRP-Dataset-v1)

## 1. Summary and Scope
- **Dataset Name:** RRP Historical Release Defect Benchmark
- **Target Task:** Predicting whether a proposed software change or release will have at least one qualifying defect linked to it within a 30-day post-release observation window.
- **Unit of Prediction:** Repository commit / merged pull request.
- **Language Scope:** Python software projects.

## 2. Labeling Methodology & Definitions
- **Positive Class (`1`):** A qualifying defect report (issue labeled bug/defect or linked fix commit) is linked to the change and observed within a 30-day follow-up window.
- **Negative Class (`0`):** The 30-day follow-up window has fully elapsed, the issue tracker was consistently active, and no qualifying defect was linked.
- **Unknown / Excluded:** Commit changes where defect attribution is ambiguous, or where the 30-day follow-up window has not elapsed.

## 3. Leakage Guardrails
- **No Future Information:** Post-release bug reports, subsequent bug-fix commits, or post-deployment test reports are strictly excluded from input features ($X$).
- **Prediction-Time Only:** All features (cyclomatic complexity, lines of code, churn, previous defect history, and pre-release test failure ratio) reflect the state of the repository *before* the proposed release or commit merge.

## 4. Feature Schema
| Feature Name | Type | Description |
|---|---|---|
| `files_changed` | Integer | Count of modified files in the release/commit |
| `lines_added` | Integer | Lines of code added |
| `lines_deleted` | Integer | Lines of code deleted |
| `total_churn` | Integer | Total churn (`lines_added + lines_deleted`) |
| `cyclomatic_complexity` | Float | Sum of cyclomatic complexities across modified blocks |
| `maintainability_index` | Float | Radon Maintainability Index (0–100 scale) |
| `loc` | Integer | Total lines of code in the project snapshot |
| `function_count` | Integer | Count of functions in the project snapshot |
| `recent_commit_frequency` | Integer | Commits in the repository within the preceding 30 days |
| `previous_defects_count` | Integer | Cumulative historical defects prior to this commit |
| `test_failure_ratio` | Float | Fraction of pre-release tests/checks that failed |

## 5. Splitting Strategy
- **Chronological Split:** Commits are ordered strictly by timestamp.
- **Training Set (70%):** Earliest changes.
- **Validation Set (15%):** Intermediate period for hyperparameter tuning and threshold selection.
- **Test Set (15%):** Latest untouched period for final unbiased evaluation.

## 6. Known Limitations
- Does not generalize directly to compiled languages (C++, Rust) or projects without explicit bug tracking.
- Absence of recorded bug does not guarantee 100% bug-free software (reporting bias).
- Missing CI or test data must be flagged as `unknown` rather than imputed as 0 risk.
