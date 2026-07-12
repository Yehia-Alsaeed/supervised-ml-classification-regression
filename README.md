# Supervised ML Classification and Regression

This repository contains three fully functional machine learning pipelines for supervised learning across classification and regression tasks. The pipelines cover end-to-end model development: data inspection, cleaning, exploratory visualization, preprocessing, feature engineering, feature selection, hyperparameter tuning, learning curves, and final model evaluation.

The project is organized as a compact portfolio repository that demonstrates how different machine learning algorithms behave across tabular datasets with different targets and data-quality challenges.

## Tech Stack & Core Skills

- **Languages & Frameworks:** Python, pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning Techniques:** Logistic Regression, Random Forest, Decision Trees, KNN, Linear Regression
- **Data Engineering:** Reusable Preprocessing Pipelines, Imputation, Scaling, One-Hot Encoding, Target Encoding
- **Model Optimization:** `GridSearchCV` hyperparameter tuning, `SelectKBest` feature selection
- **Evaluation:** ROC AUC, F1-score, Precision, Recall, Accuracy, R2, RMSE, MAE, Learning Curves

## Methodology & Core Projects

The repository applies a rigorous, structured methodology across all tasks, utilizing stratified train/test splits, robust preprocessing pipelines, and learning curves to evaluate underfitting, overfitting, and generalization behavior.

| Pipeline | Task | Goal | Main Models |
| --- | --- | --- | --- |
| `adult_income_classification.ipynb` | Binary classification | Predict whether annual income is above 50K | Logistic Regression, Random Forest, KNN |
| `car_insurance_classification.ipynb` | Binary classification | Predict whether a customer buys car insurance | Logistic Regression, Decision Tree, KNN |
| `bias_correction_regression.ipynb` | Regression | Predict next-day maximum temperature | Linear Regression, Decision Tree Regressor, Random Forest Regressor |

## Results & Visuals

Detailed result tables are available in [docs/results.md](docs/results.md).

![Shared Machine Learning Workflow](docs/assets/ml-workflow.svg)

### Adult Income Classification

The pipeline predicts whether a person earns more than 50K per year. After cleaning, duplicate removal, target encoding, and preprocessing, three classifiers were compared.

**Best Result: Tuned Random Forest**
- Accuracy: 0.855
- ROC AUC: 0.909
- Strongest overall ranking performance among the tested models

![Adult Income Classification Results](docs/assets/adult-income-results.svg)

### Car Insurance Classification

The pipeline predicts whether a customer is likely to buy car insurance. The workflow includes call-duration feature engineering, missing-value handling, artificial missing-value simulation, feature selection, and tuned classification pipelines.

**Best Balanced Model: Tuned Logistic Regression**
- Accuracy: 0.831
- F1-score for the positive class: 0.793
- ROC AUC: 0.892
- Best overall balance between precision, recall, interpretability, and ranking performance

![Car Insurance Classification Results](docs/assets/car-insurance-results.svg)

### Bias Correction Regression

The pipeline predicts next-day maximum temperature from weather-station features. The workflow includes date handling, missing-value imputation, outlier clipping, statistical feature selection, model tuning, and regression diagnostics.

**Best Result: Tuned Random Forest Regressor**
- R2: 0.881
- RMSE: 1.047
- MAE: 0.783
- Best model among Linear Regression, Decision Tree, and Random Forest

![Bias Correction Regression Results](docs/assets/bias-correction-results.svg)

## Repository Structure

```text
adult_income_classification.ipynb      Adult income classification pipeline
car_insurance_classification.ipynb     Car insurance classification pipeline
bias_correction_regression.ipynb       Temperature bias-correction regression pipeline
docs/results.md                        Compact model comparison tables
docs/assets/                           Result charts and workflow diagram
requirements.txt                       Python dependencies
```

## Requirements

If you wish to run these pipelines locally:
- Python dependencies can be installed via `pip install -r requirements.txt`.
- Datasets are not included in this repository. You must acquire the `adult.csv`, `carInsurance_train.csv`, and `Bias_correction_ucl.csv` datasets and place them in a local `data/` folder before running the notebooks.
