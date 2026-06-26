# Supervised ML Classification and Regression

This repository contains three Jupyter Notebook workflows for supervised machine learning across classification and regression tasks. The notebooks cover end-to-end model development: data inspection, cleaning, exploratory visualization, preprocessing, feature engineering, feature selection, hyperparameter tuning, learning curves, and final model evaluation.

The project is organized as a compact portfolio repository that demonstrates how different machine learning algorithms behave across tabular datasets with different targets and data-quality challenges.

## Notebooks

| Notebook | Task | Goal | Main Models |
| --- | --- | --- | --- |
| `adult_income_classification.ipynb` | Binary classification | Predict whether annual income is above 50K | Logistic Regression, Random Forest, KNN |
| `car_insurance_classification.ipynb` | Binary classification | Predict whether a customer buys car insurance | Logistic Regression, Decision Tree, KNN |
| `bias_correction_regression.ipynb` | Regression | Predict next-day maximum temperature | Linear Regression, Decision Tree Regressor, Random Forest Regressor |

## Project Highlights

- Built reusable preprocessing pipelines with imputation, scaling, and one-hot encoding.
- Used stratified train/test splits for classification tasks.
- Applied feature selection with `SelectKBest` for classification and regression workflows.
- Tuned model hyperparameters with `GridSearchCV`.
- Compared models using classification metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
- Compared regression models using R2, MSE, RMSE, and MAE.
- Used learning curves to evaluate underfitting, overfitting, and generalization behavior.

## Results Summary

### Adult Income Classification

The Adult Income notebook predicts whether a person earns more than 50K per year. After cleaning, duplicate removal, target encoding, and preprocessing, three classifiers were compared.

Best result:

- Tuned Random Forest
- Accuracy: 0.855
- ROC AUC: 0.909
- Strongest overall ranking performance among the tested models

### Car Insurance Classification

The Car Insurance notebook predicts whether a customer is likely to buy car insurance. The workflow includes call-duration feature engineering, missing-value handling, artificial missing-value simulation, feature selection, and tuned classification pipelines.

Best balanced model:

- Tuned Logistic Regression
- Accuracy: 0.831
- F1-score for the positive class: 0.793
- ROC AUC: 0.892
- Best overall balance between precision, recall, interpretability, and ranking performance

### Bias Correction Regression

The Bias Correction notebook predicts next-day maximum temperature from weather-station features. The workflow includes date handling, missing-value imputation, outlier clipping, statistical feature selection, model tuning, and regression diagnostics.

Best result:

- Tuned Random Forest Regressor
- R2: 0.881
- RMSE: 1.047
- MAE: 0.783
- Best model among Linear Regression, Decision Tree, and Random Forest

## Results and Workflow

Detailed result tables are available in [docs/results.md](docs/results.md).

![Shared Machine Learning Workflow](docs/assets/ml-workflow.svg)

![Adult Income Classification Results](docs/assets/adult-income-results.svg)

![Car Insurance Classification Results](docs/assets/car-insurance-results.svg)

![Bias Correction Regression Results](docs/assets/bias-correction-results.svg)

## Repository Structure

```text
adult_income_classification.ipynb      Adult income classification workflow
car_insurance_classification.ipynb     Car insurance classification workflow
bias_correction_regression.ipynb       Temperature bias-correction regression workflow
docs/results.md                        Compact model comparison tables
docs/assets/                           Result charts and workflow diagram
requirements.txt                       Python dependencies
```

## Data Setup

The datasets are not included in this repository. To rerun the notebooks locally, create a `data/` folder and place the dataset files using these names:

```text
data/adult.csv
data/carInsurance_train.csv
data/Bias_correction_ucl.csv
```

The notebooks use these relative paths so they can run cleanly on any machine after the dataset files are added locally.

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
