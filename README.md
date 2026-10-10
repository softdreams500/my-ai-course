# AI and data science course projects

Learning notebooks by David Solomon, organized by project. Existing datasets, notebooks, saved outputs, and file names are retained.

| Project | Entry point | Input |
| --- | --- | --- |
| Diabetes exploration and rule-based risk modeling | [Project README](<Diabetes Risk Prediction (EDA on CDC data)/README.md>) | `diabetes_sample.csv` in that directory |
| A/B conversion analysis | [Project README](ab_testing/README_AB_Testing_Project.md) | `ab_testing/ab_test_results.csv` |
| Retail customer segmentation | [Project README](retail_segmentation/README.md) | `retail_segmentation/retail_customers_clean.csv` |
| Nigeria excess-death trends | [Separate repository](https://github.com/softdreams500/covid-nigeria-trends) | Existing `covid-trend` submodule |

## Setup

Requires Python 3. Create a virtual environment and install the direct notebook dependencies:

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m jupyter lab
```

On macOS/Linux use `.venv/bin/python` instead. Select the virtual environment's Python kernel. Run each notebook with its project directory as the working directory, then restart the kernel and run all cells in order.

For the COVID project, run `git submodule update --init covid-trend` and follow its own README and dependency file. The submodule stays pinned to the existing commit.

Dependencies are unpinned; this is a starting environment, not a verified reproducibility lock. Historical notebook outputs remain in place and may precede source fixes. Review fresh results before quoting metrics.

## Review notes

The diabetes notebook's fallback target is computed from input features. Its scores measure how well the model reproduces that rule, rather than independently validated diabetes prediction. The empty `outputs/figures/cleaned data.csv` is an existing placeholder. Duplicate sample CSV files are retained. See each project's README for the actual committed inputs and outputs.
