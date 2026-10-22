# Iris Flower Classification: Comparing Machine Learning Models

A hands-on Machine Learning coursework project exploring an end-to-end classification workflow on the Iris dataset. The project compares **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes** using test-set accuracy and recall.

## Project Objective

The goal is to understand how different classification algorithms perform on the same dataset and compare their predictions using consistent evaluation metrics.

## Dataset

- **Source:** Iris dataset downloaded from Kaggle.
- **Original size:** 150 rows and 5 columns.
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
- **Target:** `species` (Setosa, Versicolor, Virginica).
- The current cleaning notebook removes exact duplicate rows, leaving **147 rows** for the subsequent split.

> Note: Exact duplicate observations are not necessarily erroneous. Removing them changes the dataset and its class counts, so this is a methodological choice rather than an automatic cleaning requirement.

## Workflow

1. **Data analysis** — inspect the dataset, columns, data types, summary statistics, and shape.
2. **Data cleaning** — check for duplicate records, standardize species labels, and replace negative measurement values with missing values.
3. **Train-test split** — separate the features and target, then split the cleaned data into 80% training and 20% testing sets using `random_state=42`.
4. **Preprocessing review** — examine whether missing-value handling, scaling, encoding, feature selection, and transformations are needed for this dataset.
5. **Model training and prediction** — fit the three classification approaches and predict species for the test set.
6. **Evaluation** — compare test accuracy, class-wise recall, classification reports, and confusion matrices where calculated.

## Models

### 1. Logistic Regression
A multiclass Logistic Regression classifier implemented with scikit-learn using `max_iter=1000`.

### 2. K-Nearest Neighbors (KNN)
The notebook first evaluates a KNN classifier with `n_neighbors=5`. It also contains a `Pipeline` combining `StandardScaler` and `KNeighborsClassifier`, with `GridSearchCV` testing `k` values of 3, 5, 7, and 9 using 5-fold cross-validation and accuracy as the selection metric.

After updating the train-test split to use `stratify=y` and rerunning the notebook, the pipeline's best parameter is `k=3`, with a mean cross-validation accuracy of approximately `0.9489`.

### 3. Gaussian Naive Bayes
A Gaussian Naive Bayes classifier implemented using scikit-learn's `GaussianNB`.

## Recorded Test Results

The following results are from the notebooks after the train-test split was updated to use `stratify=y` and the notebooks were rerun. The test set contains 30 observations (10 from each Iris species).

| Model / experiment | Test accuracy | Class-wise recall (Setosa, Versicolor, Virginica) |
|---|---:|---|
| Logistic Regression | 96.67% | 1.00, 0.90, 1.00 |
| KNN (`k=5`, without scaling) | 100.00% | 1.00, 1.00, 1.00 |
| Gaussian Naive Bayes | 93.33% | 1.00, 0.90, 0.90 |
| KNN Pipeline + GridSearchCV (`k=3`) | 93.33% | Not shown in the saved output |

**KNN Pipeline note:** With the stratified split, `GridSearchCV` selected `k=3`, with a mean cross-validation accuracy of approximately `0.9489`. The pipeline's test accuracy was `93.33%`. The saved output does not include the pipeline's class-wise recall or classification report, so those values are not reported here.

Among the results shown, the unscaled KNN model with `k=5` achieved `100.00%` test accuracy on this particular 30-observation test set. Logistic Regression achieved `96.67%`, while Gaussian Naive Bayes and the tuned KNN Pipeline each achieved `93.33%`. These results describe this specific split and setup only; they do not establish that one approach will always perform best on other data or splits. The KNN `k=5` experiment and the scaled, tuned KNN Pipeline are separate configurations and should not be treated as the same experiment.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib / Seaborn (where used in analysis)
- scikit-learn
- Git and GitHub

## Repository Structure

```text
Machine-Learning/
├── IRIS_Dataset.csv
├── IRIS_Data_Analysis.ipynb
├── IRIS_Data_Cleaning.ipynb
├── IRIS_Data_Spliting.ipynb
├── Data_Preprocess_And_Feature_Engineer.ipynb
├── Logistic_Regression.ipynb
├── K_Nearest_Neighbors_KNN.ipynb
├── Naive_Byes.ipynb
└── README.md
```

## Running the Notebooks

1. Clone or download this repository.
2. Create and activate a Python environment.
3. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn notebook ipynb
   ```

4. Open the project in Jupyter Notebook or VS Code.
5. Keep all notebooks and `IRIS_Dataset.csv` in the same project directory.
6. Run the notebooks in workflow order: analysis, cleaning, splitting, preprocessing, then model notebooks.

The current notebooks import variables from other notebooks using `ipynb.fs.full`. This means execution depends on the `ipynb` package and on running the notebooks in a compatible environment. If imports fail, consider refactoring shared data-preparation steps into a Python module or loading the saved dataset and repeating the split consistently in each model notebook.

## Limitations and Next Steps

- Revisit whether removing exact duplicate rows is appropriate for the assignment and document the rationale.
- Use a stratified train-test split for class-proportion preservation, especially when comparing classification metrics.
- Keep preprocessing steps that learn parameters (such as scaling or imputation) inside a scikit-learn `Pipeline`, fitted only on training data.
- Report macro recall or weighted recall in addition to class-wise recall when a single summary score is needed.
- Print a full evaluation report for the KNN pipeline after hyperparameter selection.
- Consider using cross-validation consistently for model comparison and avoid drawing broad conclusions from one test split.

## Learning Takeaway

This project provided practice with data inspection, cleaning, train-test splitting, classification model implementation, hyperparameter selection, and evaluation using accuracy and recall. It is part of an ongoing learning journey in Machine Learning.