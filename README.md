# Automated Loan Approval Prediction System

An end-to-end machine learning system designed to evaluate loan eligibility using predictive financial metrics, historical applicant behaviors, and multi-category asset profiles. Utilizing an optimized ensemble learning architecture, this system provides credit risk classification performance to streamline institutional lending workflows.

---

## Project Architecture & Pipeline

The system implements a structured, modular machine learning pipeline traversing the complete lifecycle from ingestion to interactive simulation:

[ Raw Ingestion ] ➔ [ Structural Audit ] ➔ [ Label Encoding ] ➔ [ Stratified Split ] ➔ [ Random Forest Training ] ➔ [ Serialized Deployment ]


1. **Data Ingestion & Structural Cleaning:** Automatic parsing of tabular source files coupled with automated string trimming to eliminate hidden whitespace from feature headers.
2. **Exploratory Data Analysis (EDA):** Statistical profiling and data visualization to isolate high-impact indicators, confirming credit rating (CIBIL score) as a primary feature split vector.
3. **Data Preprocessing & Feature Engineering:** Mapping categorical text strings (`education`, `self_employed`, `loan_status`) into structured binary integers (`0` and `1`) via automated Label Encoding.
4. **Data Partitioning:** Implementation of a stratified 80/20 train-test split to ensure class balance across subsets.
5. **Model Engineering:** Training and optimization of an ensemble classification model.
6. **Artifact Preservation & Deployment:** Serialization of model weights to local storage and deployment of an internal prediction interface.

---

## Model Evaluation Metrics

The system was validated using a stratified test subset containing 854 records to ensure objective evaluation of model generalization.

### Final Classification Accuracy: 98.36%

```text
Detailed Classification Report:

              precision    recall  f1-score   support

 0 (Approved)       0.98      0.99      0.99       531
 1 (Rejected)       0.99      0.97      0.98       323

    accuracy                           0.98       854
   macro avg       0.99      0.98      0.98       854
weighted avg       0.98      0.98      0.98       854
Performance Analysis
Precision (98% - 99%): Out of all credit applications flagged by the model as low risk, 98% to 99% were correctly classified, significantly mitigating institutional credit risk.

Recall (97% - 99%): The model successfully captured up to 99% of qualified applicants, optimizing business fulfillment metrics.

F1-Score Symmetry: Highly balanced values across both classes confirm the structural stability of the model, establishing that predictions are not skewed by class imbalances.

Technical Stack & Key Libraries
Environment: Google Colab / Jupyter Notebooks

Data Core: pandas, numpy

Visualization Layer: seaborn, matplotlib

Machine Learning Framework: scikit-learn

Model Serialization: joblib

Repository Structure
LoanApproval.ipynb — Primary Jupyter Notebook containing data exploration, preprocessing, and training pipelines.

loan_approval_dataset.csv — Tabular financial and asset dataset.

loan_prediction_model.pkl — Serialized, production-ready model artifact containing optimal tree structures.

README.md — Core technical documentation.

Monolithic Implementation Script
Below is the complete, verified Python implementation mapping to the steps outlined above. This script can be executed inside a single development window or across sequential cells within the notebook environment:

Python
# ==============================================================================
# PROJECT: AUTOMATED LOAN APPROVAL PREDICTION SYSTEM
# ENVIRONMENT: Google Colab / Jupyter Notebook
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: ENVIRONMENT SETUP & DATA INGESTION
# ------------------------------------------------------------------------------
print("Executing Phase 1: Environment Setup & Data Ingestion...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configure inline visualization parameters
sns.set_theme(style="whitegrid")

# Load the locally uploaded dataset files
try:
    df = pd.read_csv('loan_approval_dataset.csv')
    # Strip hidden leading or trailing spaces from column headers
    df.columns = df.columns.str.strip()
    print("Dataset ingested successfully.")
    print(f"Data Matrix Dimensions: {df.shape[0]} records, {df.shape[1]} features\n")
except FileNotFoundError:
    print("Error: 'loan_approval_dataset.csv' not found in local workspace directory.")
    print("Please upload the file to the Colab sidebar panel and re-run.")

# Display initial layout preview
print("First 5 rows of raw dataset:")
print(df.head())


# ------------------------------------------------------------------------------
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
print("\n" + "="*80 + "\n")
print("Executing Phase 2: Exploratory Data Analysis (EDA)...")

# Execute comprehensive structural summary audit
print("\n--- Structural Framework Audit (df.info()) ---")
df.info()

# Render credit validation parameter metrics comparison
print("\nRendering Statistical Exploratory Visualizations...")
plt.figure(figsize=(9, 5))
sns.boxplot(x='loan_status', y='cibil_score', data=df, palette='muted')
plt.title('Impact of CIBIL Score on Loan Approval Status', fontsize=14, fontweight='bold')
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('CIBIL Score', fontsize=12)
plt.show()


# ------------------------------------------------------------------------------
# PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
print("\n" + "="*80 + "\n")
print("Executing Phase 3: Data Preprocessing & Feature Engineering...")

# Initialize the structural transformer mapping tool
le = LabelEncoder()

# Transform categorical text string metrics to categorical numerical values
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])
df['loan_status'] = le.fit_transform(df['loan_status'])
print("Categorical parameters securely mapped to binary vector values.")

# Isolate evaluation predictors (X) from objective metrics labels (y)
X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']
print("Features (X) and Target Vector (y) split completed successfully.")

# Execute stratified train-test operational partition
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training Sets Dimensions: Features {X_train.shape} | Targets {y_train.shape}")
print(f"Testing Sets Dimensions:  Features {X_test.shape}  | Targets {y_test.shape}")


# ------------------------------------------------------------------------------
# PHASE 4: MODEL TRAINING & EVALUATION
# ------------------------------------------------------------------------------
print("\n" + "="*80 + "\n")
print("Executing Phase 4: Model Training & Evaluation...")

# Initialize optimized ensemble random forest tree configurations
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit weights optimizations vectors to input matrices datasets
model.fit(X_train, y_train)
print("Ensemble classification training matrix calculations complete.")

# Map evaluations vectors predictions across test parameters
y_pred = model.predict(X_test)

# Print high-level classification system accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Verified Model Execution Accuracy: {accuracy * 100:.2f}%")

# Generate granular production tracking report classification parameters
print("\nDetailed Production Classification Report Parameters:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------------------------
# PHASE 5: SERIALIZATION & PERSISTENCE
# ------------------------------------------------------------------------------
print("\n" + "="*80 + "\n")
print("Executing Phase 5: Model Serialization & Persistence...")

# Save operational weights model tracking structure parameters files to workspace
joblib.dump(model, 'loan_prediction_model.pkl')
print("Production binaries file artifact saved successfully as: 'loan_prediction_model.pkl'")


# ------------------------------------------------------------------------------
# INTERACTIVE PRODUCTION TESTING MODULE (COLAB FORMS COMPONENT)
# ------------------------------------------------------------------------------
#@title 🏦 Loan Eligibility Interactive Tester { run: "auto" }

# This subsection auto-reloads whenever input forms sliders are modified
import pandas as pd
import joblib

try:
    # Read the serialized compiled weights file back into application framework
    loaded_model = joblib.load('loan_prediction_model.pkl')
    
    # Configuration inputs form metrics panels parameters UI
    no_of_dependents = 2 #@param {type:"slider", min:0, max:5, step:1}
    education = "Graduate" #@param ["Graduate", "Not Graduate"]
    self_employed = "No" #@param ["No", "Yes"]
    income_annum = 5000000 #@param {type:"number"}
    loan_amount = 15000000 #@param {type:"number"}
    loan_term = 12 #@param {type:"slider", min:2, max:20, step:2}
    cibil_score = 750 #@param {type:"slider", min:300, max:900, step:10}
    residential_assets_value = 4000000 #@param {type:"number"}
    commercial_assets_value = 2000000 #@param {type:"number"}
    luxury_assets_value = 6000000 #@param {type:"number"}
    bank_asset_value = 3000000 #@param {type:"number"}

    # Reverse-engineering manual string form fields inputs maps to internal matrix configurations
    edu_encoded = 0 if education == "Graduate" else 1
    emp_encoded = 1 if self_employed == "Yes" else 0

    # Package arguments parameters within a formal structural dataframe array to bypass validation errors
    input_data = pd.DataFrame([[
        no_of_dependents, edu_encoded, emp_encoded, income_annum,
        loan_amount, loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value
    ]], columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])

    # Evaluate dynamic validation predictions across real-time parameters
    prediction = loaded_model.predict(input_data)

    print("\n" + "="*48)
    print("          BANK ENGINE EVALUATION OUTPUT         ")
    print("="*48)
    if prediction[0] == 0:
        print("📈 STATUS: LOAN APPLICATION APPROVED")
        print("Basis: Applicant satisfies structural collateral assets parameters.")
    else:
        print("❌ STATUS: LOAN APPLICATION REJECTED")
        print("Basis: High default risk classification index threshold tripped.")
    print("="*48)

except FileNotFoundError:
    print("\nSystem Warning: Complete pipeline logic execution loops prior to evaluating tester interface modules.")

Future Enhancements
Microservices Deployment: Wrapping the serialized .pkl binary file within a standalone Flask or Streamlit micro-framework API.

Interface Optimization: Designing a dark-themed user interface utilizing a high-contrast palette of slate blue backgrounds and light grey inputs to emulate banking terminal application design.
