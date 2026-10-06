AL/ML PROJECT - SVM INDIVIDUAL PART
=================================

FILES
-----
SVM.ipynb       -> Google Colab notebook for the SVM part
SVM.py          -> Same SVM implementation as a Python script
data/           -> Prepared scaled and PCA train/test CSV files
outputs/        -> Output folder
SVM_Report.pdf  -> Report for the SVM individual part

GOOGLE COLAB
------------
1. Upload the AL_ML_PROJECT_SVM folder to MyDrive.
2. Open SVM.ipynb with Google Colab.
3. Run the first cell and allow Google Drive access.
4. The notebook uses this default data path:
   /content/drive/MyDrive/AL_ML_PROJECT_SVM/data
5. Run the cells from top to bottom.

SVM WORKFLOW
------------
1. Load the prepared scaled and PCA datasets.
2. Separate input features and target labels.
3. Train Linear, RBF and Polynomial SVM models.
4. Compare RBF on scaled data and PCA data.
5. Tune RBF with a small GridSearchCV.
6. Evaluate the best model using accuracy and a classification report.
7. Display a confusion matrix and model comparison chart.
8. Save the best model as best_svm_model.pkl.

VERIFIED RESULTS ON THE INCLUDED DATA
-------------------------------------
Linear SVM       : 83.88%
RBF SVM          : 93.71%
Polynomial SVM   : 88.30%
RBF + PCA        : 91.22%
Tuned RBF SVM    : 97.24%
Best parameters  : C=10, gamma=0.1, kernel=rbf
