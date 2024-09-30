

# 🛡️ A Comprehensive Study on Enhancing Classification Performance with Oversampling and Undersampling in Credit Card Fraud Data

**Author**: Usha  
**Platform**: Entry Elevate  
**Date**: 25-09-2024  
**Project**: Final Capstone Project in Machine Learning from Entry Elevate  
**Domain**:cybersecurity
![image](https://github.com/user-attachments/assets/87ee64fc-86f4-4e56-b4e4-e0a3427a3dc4)


Welcome to the repository of my **Final Capstone Project in Machine Learning** from Entry Elevate. This project explores how different techniques for handling class imbalance, such as **oversampling** and **undersampling**, affect the performance of machine learning models in detecting credit card fraud. The study aims to provide insights into which sampling techniques work best for fraud detection models.

---

## 📋 Project Overview

In fraud detection, the **class imbalance** between the majority class (non-fraudulent transactions) and the minority class (fraudulent transactions) poses a significant challenge to machine learning models. This project focuses on addressing this imbalance using two popular techniques:

- **Undersampling**: Reducing the size of the majority class to match the minority.
- **Oversampling**: Using **Synthetic Minority Over-sampling Technique (SMOTE)** to generate synthetic examples for the minority class.

By evaluating how these techniques affect the performance of various classification models, the project aims to improve the detection of fraudulent transactions while minimizing false negatives.

---

## 📂 Dataset

- **Source**: Real-world **Credit Card Fraud Detection Dataset**
- [https://data.world/vlad/credit-card-fraud-detection](https://data.world/vlad/credit-card-fraud-detection/workspace/file?filename=CC.csv)
- **Records**: 284,807 transactions
- **Features**: 30 anonymized variables (V1-V28), along with 'Amount' and 'Time'
- **Target Variable**: 'Class' (0 = Non-fraud, 1 = Fraud)
  
This highly imbalanced dataset contains only about 0.17% fraudulent transactions, making it challenging to achieve high performance with standard machine learning models without proper data balancing techniques.

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- **Feature Selection**: After exploring correlations, selected significant features that contribute to fraud detection.
- **Data Scaling**: Applied **standardization** (excluding 'Time') to bring features onto a common scale.
- **Outlier Removal**: Identified and removed outliers that could distort model training.
- **Train-Test Split**: Split the dataset into training and testing sets for validation purposes.

### 2. Handling Class Imbalance
To address the severe class imbalance, the following techniques were employed:
- **Undersampling**: Reduced the size of the majority class (non-fraudulent transactions) to match the number of minority class samples (fraudulent transactions).
- **Oversampling (SMOTE)**: Generated synthetic samples for the minority class using the **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the dataset.

### 3. Model Selection
The following machine learning models were used to classify transactions:
- **Logistic Regression**: A linear model suitable for binary classification tasks.
- **Decision Tree**: A non-linear model that creates a tree-like structure for decision making.
- **Random Forest**: An ensemble of decision trees, improving robustness and accuracy.
- **svm**
- **k-nearest neighbour**
### 4. Model Evaluation Metrics
To thoroughly evaluate the performance of the models, multiple metrics were used:
- **Accuracy**: Measures overall model performance.
- **Precision, Recall, F1-Score**: Focus on the minority class (fraud detection) to assess how well the model handles fraud.
- **Confusion Matrix**: Provides a breakdown of true positives, false positives, true negatives, and false negatives.
  
These metrics helped to determine how well the models balanced between detecting fraud and minimizing false positives.

---

## 📊 Model Accuracy Comparison: Undersampling vs. Oversampling

### 1. **Undersampling Outperforms Oversampling Across All Models**
For all three machine learning models (**Logistic Regression**, **Decision Tree**, and **Random Forest**), **undersampling** consistently achieves higher accuracy compared to **oversampling**.

### 2. **Decision Tree Model Shows Smallest Gap Between Undersampling and Oversampling**
The **Decision Tree** model exhibits the smallest difference in accuracy between undersampling and oversampling, suggesting that it is less sensitive to the sampling technique.

### 3. **Random Forest with Undersampling Achieves Highest Accuracy**
**Random Forest** with **undersampling** reaches a perfect accuracy of **1.00**. However, this may indicate overfitting due to the imbalanced dataset. While Random Forest handles the training data well, it may not generalize effectively to unseen data.

### 4. **Logistic Regression Shows Most Significant Difference Between Undersampling and Oversampling**
**Logistic Regression** shows the largest accuracy gap between undersampling (0.93) and oversampling (0.97), highlighting the importance of choosing the appropriate sampling method.

---

## 🛠 Challenges of Working with Imbalanced Datasets

When dealing with imbalanced datasets, the choice between undersampling and oversampling methods presents several challenges. Below are the key implications of each approach and the need for effective communication with stakeholders.

### 1. **Oversampling Challenges**
While oversampling can improve model performance (particularly in terms of **recall** and **F1-score**), it has notable drawbacks:
- **Increased Training Time**: Oversampling often results in larger datasets, as seen in our experiments with over 200,000 records. This can lead to significantly longer training times and increased computational costs.
- **Risk of Overfitting**: By duplicating existing minority class samples, oversampling may cause models to overfit, learning patterns specific to the training data that do not generalize well.
- **High Accuracy Masking Poor Performance**: Models trained on oversampled data may achieve high accuracy (97-100%), but such figures can be misleading, as they may perform poorly on the minority class (fraud detection).

### 2. **Undersampling Challenges**
Undersampling reduces dataset size, but it has its own set of challenges:
- **Loss of Information**: The primary concern with undersampling is the loss of potentially valuable data, as a significant portion of the majority class is discarded.
- **Unbalanced Data**: Even with a smaller dataset, some imbalance may persist, causing the model to struggle with detecting the minority class.

### 3. **Communicating with Stakeholders**
It is crucial to communicate the challenges and trade-offs of each approach to stakeholders clearly. Important discussion points include:
- **Data Quality and Quantity**: Highlight the importance of collecting a more balanced dataset to improve model performance.
- **Model Performance Metrics**: Explain that **accuracy** alone is insufficient and should be complemented with metrics like **precision, recall**, and **F1-score**, especially for fraud detection.
- **Potential Solutions**: Recommend alternative strategies:
  - **SMOTE**: Generates synthetic samples instead of duplicating existing ones.
  - **Ensemble Methods**: Use multiple models to capture different data aspects.
  - **Cost-sensitive Learning**: Assign higher costs to misclassifying the minority class.

---

## 📝 Key Findings

- **Undersampling** consistently outperforms **oversampling** across models, but comes with the risk of losing valuable data.
- **Random Forest** with **undersampling** achieved the highest accuracy, but the model may overfit the imbalanced data.
- High accuracy does not guarantee the model effectively detects the minority class, emphasizing the need for balanced evaluation using **precision**, **recall**, and **F1-score**.

---

## 🎯 Conclusion

This project highlights the critical role that **class imbalance handling** plays in improving machine learning models for fraud detection. **Undersampling** delivered better results in this case, but the risks of losing important data must be considered. By experimenting with both **oversampling** and **undersampling**, and evaluating models based on multiple metrics, we can develop more effective fraud detection solutions.

---

## 💡 Recommendations

1. **Use Undersampling with Caution**: While effective, it can result in a loss of crucial data, especially in larger datasets.
2. **Hybrid Techniques**: Consider combining **SMOTE** with **Tomek Links** to retain important minority class samples while reducing majority class noise.
3. **Comprehensive Model Evaluation**: Ensure that accuracy is not the sole measure of success. Use **precision, recall**, and **F1-score** to get a more holistic view of model performance.
4. **Future Exploration**: Investigate advanced techniques like **XGBoost** and **Gradient Boosting** for improved handling of imbalanced datasets.

---

