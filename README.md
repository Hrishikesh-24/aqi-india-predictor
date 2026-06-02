# AQI Predictor and Pollution Analyzer

This project uses the India AQI dataset to build:

1. AQI prediction using regression models
2. Pollution level prediction using classification models
3. Pollution pattern analysis using K-Means clustering
4. A Streamlit UI with separate prediction output for each model
5. An India city map showing AQI levels

## Dataset

Default dataset path used in the code:

```text
C:\Users\HP\AQi-Project\INDIA_AQI_COMPLETE_20251126.csv
```

The inspected dataset has 842,160 rows, 71 columns, and 29 Indian cities.

## Setup

Open PowerShell in this folder and run:

```powershell
pip install -r requirements.txt
```

If `python` or `pip` is not available, install Anaconda or use your Jupyter environment package manager.

## Step 1: Use the Jupyter notebook

Open:

```text
AQI_Project_Notebook.ipynb
```

Run cells from top to bottom. The notebook covers:

- Loading the dataset
- Cleaning missing values
- Creating AQI category labels
- Training regression models
- Training classification models
- Running K-Means clustering
- Saving trained models
- Launching Streamlit

## Step 2: Run the Streamlit UI

From this folder:

```powershell
streamlit run app.py
```

The app includes:

- Dashboard and India map
- Separate regression predictions from Linear Regression, Decision Tree, and Random Forest
- Separate classification predictions from Logistic Regression, Decision Tree, and Random Forest
- K-Means pollution pattern clustering

## Models Used

Regression:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Classification:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Clustering:

- K-Means

