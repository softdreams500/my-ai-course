# Retail customer segmentation

A learning project using standardization, K-Means with four clusters, and PCA to explore customer groups.

## Files

| File | Role |
| --- | --- |
| `retail_customers_clean.csv` | Committed notebook input |
| `retail_segmentation.ipynb` | Exploration, scaling, elbow plot, clustering, PCA |
| `retail_customer_segments.csv` | Saved customer assignments and PCA coordinates |

## Run

Install the repository root's `requirements.txt`. Start Jupyter with this directory as the working directory, open `retail_segmentation.ipynb`, and run all cells from a fresh kernel. The final cell overwrites `retail_customer_segments.csv`.

The notebook uses age, encoded gender, annual income, spending score, purchases per year, and loyalty years. It tries k=1 through 10 and then selects k=4. Cluster labels are arbitrary; interpret them using the group summaries. PCA is a visualization, and proposed marketing actions need separate validation. The earlier README described synthetic data; verify provenance before external use.

Potential follow-up work: evaluate silhouette scores, document the choice of k, and explicitly record dependency versions. The original datasets and notebook outputs remain available.
