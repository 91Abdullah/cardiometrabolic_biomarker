
# Non-Invasive Cardiometabolic Biomarker Estimation Using Retinal Fundus Images

Diagnosing noncommunicable diseases like cardiovascular disease and diabetes typically requires invasive, costly, and time-consuming blood sample analyses. This project leverages deep learning to address these limitations by estimating common cardiometabolic biomarkers from non-invasive retinal fundus (RF) images.

## Project Overview

Using a dataset of 15,802 RF images from 5,653 participants in the Qatar Biobank (QBB), we developed deep-learning models to predict a wide range of cardiometabolic biomarkers. Our work includes 19 models designed to estimate biomarkers across seven health categories:

- Demographics and Body Composition
- Blood Pressure
- Lipid Profile
- Blood Profile
- Hormones
- Kidney Function
- Metabolites

Our MobileNetV3-based model demonstrated state-of-the-art performance in estimating these biomarkers for the Qatar-specific cohort, surpassing existing models in R², mean absolute error (MAE), and area under the curve (AUC) metrics. Notably, we expanded the list of predictable biomarkers from 15 to 19.

## Key Results

- **High Accuracy for Age and Gender**:
  - **Age**: MAE = 2.56, R² = 0.93
  - **Gender**: Accuracy = 96%, AUC = 0.94
- **Strong Results for Other Biomarkers**:
  - Systolic Blood Pressure: MAE = 8.02, R² = 0.49
  - Diastolic Blood Pressure: MAE = 6.06, R² = 0.45
  - Total Cholesterol: MAE = 0.63, R² = 0.29
  - LDL Cholesterol: MAE = 0.58, R² = 0.27
  - Hemoglobin: MAE = 0.79, R² = 0.60
  - Creatinine: MAE = 9.0, R² = 0.33

## Stratified Analysis

- **Gender-Stratified Performance**: Higher performance for most biomarkers in males, likely due to a higher proportion of male participants in the dataset.
- **Age-Stratified Performance**: Decline in model accuracy for older age groups.
- **Diabetes Subgroup Analysis**: Models showed better performance in the control group, highlighting the need for disease-specific models.

## Significance

This study introduces a promising method for non-invasive biomarker estimation through retinal images. This approach has the potential to revolutionize early intervention and treatment planning for noncommunicable diseases in healthcare, offering a cost-effective, non-invasive alternative to traditional blood-based biomarker measurements.

## Model and Data

- **Model Architecture**: MobileNetV3
- **Dataset**: Qatar Biobank (QBB) – 15,802 retinal fundus images from 5,653 participants.
