# Letter Recognition Group Project

Group 26-MTR-IT2011-14, IT2011. The group uses Google Colab and Jupyter notebooks.

## Project Structure

```text
Letter-Recognition-AIML/
  group_pipeline.ipynb        Complete six-member Colab pipeline
  notebooks/                  Individual preprocessing and model notebooks
  data/raw/                   Original UCI dataset
  results/eda_visualizations/  Existing figures; new group figures are generated here
  results/outputs/             Existing outputs and group_model_comparison.csv
  results/logs/               Existing log folder
```

The group notebook contains all required code in its cells. It does not import a
local `.py` module or require the repository files when uploaded to Colab.

## Members

| Student | Model Notebook | Model Branch |
|---|---|---|
| IT25102224 | notebooks/IT25102224_SVM.ipynb | SVM (8ef566e) |
| IT25102419 | notebooks/IT25102419_Decision_Tree_Letter_Recognition.ipynb | Decision-Tree (0bce57b) |
| IT25102551 | notebooks/IT25102551_Logistic_regression.ipynb | Logistic-Regression (71dafbb) |
| IT25103309 | notebooks/IT25103309_KNN.ipynb | KNN (46208d8) |
| IT25103890 | notebooks/IT25103890_Random_Forest.ipynb | Random-Forest (b71933f) |
| IT25103881 | notebooks/IT25103881_MLP.ipynb | MLP (cda3fca) |

Individual branch notebooks are preserved as source work. Original preprocessing
notebooks and original outputs remain in their existing folders.

## Colab

Upload `group_pipeline.ipynb` through Colab's File > Upload notebook menu and use
Runtime > Run all on a CPU runtime. The notebook downloads the raw UCI dataset.
The six member sections define their own models and parameter grids directly.

The common experiment removes exact duplicates, makes one stratified 80/20 split,
fits preprocessing inside five CV folds, and compares all six optimum models using
macro F1. It evaluates 196 configurations with 980 CV fits plus six final refits.
Training can take substantial time; complete it before the viva and retain the
executed notebook using File > Download > Download .ipynb.

Generated model files and tables go into `results/outputs/group/`. Generated group
figures go into `results/eda_visualizations/group/`. These folders are created when
the notebook runs. The final export cell only creates a ZIP when DOWNLOAD_OUTPUTS
is enabled. Colab's temporary files must be downloaded before the runtime ends.

The notebook includes previously verified common evaluation outputs for reference;
fresh Colab runs calculate new results. `results/outputs/group_model_comparison.csv`
is the compact recorded comparison, not a dependency for training.

## Integration Notes

KNN scaling is fold-local. SVM uses real data rather than the source notebook's dummy
fallback. Random Forest uses a documented 36-setting common grid and the shared split.
Mutual information uses discrete features before scaling; PCA follows scaling.
Outliers are audited on training data and retained rather than automatically removed.

Previously inspected test data limits claims about unseen performance. Members must
explain their own model and disclose assisted integration accurately. The final
report and assignment documents remain in the parent project folder.
