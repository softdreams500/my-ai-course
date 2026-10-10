# Diabetes data exploration and risk-rule modeling

A learning notebook that cleans sample health data, explores distributions, constructs an optional risk flag, and fits logistic regression.

## Files

- `diabetes_eda.ipynb`: exploration and modeling notebook.
- `diabetes_sample.csv`: local input dataset.
- `diabetes_risk_model.pkl`: previously saved model artifact.
- `outputs/figures/cleaned data.csv`: empty placeholder retained from the original project.

## Run

Install the dependencies from the repository root's `requirements.txt`. Open Jupyter in this directory and run `diabetes_eda.ipynb` from a fresh kernel, in cell order. The input path now uses the adjacent CSV rather than a machine-specific Windows path. Running the final cell overwrites the saved model file.

## Interpretation

When `Outcome` is absent, the notebook predicts a `High_Risk` flag derived from age/BMI or glucose. Those features also enter the model, so a high score demonstrates reproduction of the constructed rule. It does not establish independent clinical performance. Dataset provenance is not supplied with the CSV; the original CDC attribution requires verification before reuse. Historical outputs are preserved and should be regenerated before reporting results.
