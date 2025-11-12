# insurance-claims
A machine learning project to detect fraudulent insurance claims using Mendeley insurance claims dataset. Includes EDA, preprocessing, multiple model evaluations and training in the context of class imbalance. 

## Project Structure

insurance-fraud-detection/  
├── data/                     # Raw CSV   
├── src/  
│   ├── preprocessing.py      # Data cleaning and preprocessing  
│   ├── models.py             # Model training and evaluation  
├── main.py                   # Entry point for running the pipeline  
├── requirements.txt          # Environment dependencies  
├── README.md                 # Project documentation  
└── .gitignore                # Files and folders to ignore  

## Setup

1. **Clone the repository**

git clone https://github.com/jashraf215/insurance-fraud-detection.git
cd insurance-fraud-detection
2. **Create a virtual environment**
Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate

3. **Install dependencies**

pip install -r requirements.txt

4. **Add the data**

Place `insurance_claims.csv` in the `data/` folder.  
Data is found at: https://data.mendeley.com/datasets/992mh7dk9y/2 

## Running the pipeline

python main.py

This will:

- Load and clean the data  
- Train a logistic regression model  
- Print selected features and evaluation metrics  

## Outputs

- Console output: preprocessing logs, selected features, ROC AUC, average precision, confusion matrix  
- Optional saved artifacts: `final_model.joblib`, `preprocessor.joblib`, `selector.joblib`  

## Notes

- Only logistic regression is implemented as the main model as this was found to have the best model statistics
- The pipeline handles imbalanced data via class weighting  
- The code is modular: preprocessing and modeling are in `src/`, orchestrated by `main.py`

## Results

**Logistic Regression Model Performance (Test Set)**

- **ROC AUC:** 0.816  
- **Average Precision:** 0.659  

**Classification report**:

               precision    recall  f1-score   support

           0       0.90      0.76      0.83       753
           1       0.51      0.73      0.60       247

    accuracy                           0.76      1000
   macro avg       0.70      0.75      0.71      1000
weighted avg       0.80      0.76      0.77      1000
