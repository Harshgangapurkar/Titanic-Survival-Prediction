🚢 Titanic Survival Prediction with Logistic Regression
This project uses machine learning techniques to predict which passengers survived the Titanic disaster. The model is built using Logistic Regression, and the workflow covers everything from data preprocessing to evaluation using metrics like the confusion matrix and classification report.

📦 Dataset
Source: titanic.csv (loaded locally via pandas)

The dataset includes features like Sex, Age, Pclass, and other passenger-related attributes.

🧰 Tools & Libraries
Python

pandas, NumPy – for data manipulation

matplotlib, seaborn – for visualization

scikit-learn – for model evaluation (metrics like confusion matrix & classification report)

🔍 Workflow Summary
1. Initialization
Imported essential libraries (pandas, numpy, matplotlib, seaborn, sklearn)

2. Loading the Dataset

3. Data Cleaning & Feature Engineering
Checked for null values

Dropped unnecessary features

Encoded categorical variables

Handled missing values

4. Data Visualization (Insights)
Explored survival rates by gender and class using bar plots

Visualized correlations and feature distributions

5. Model Building
Applied Logistic Regression to predict survival

Used train_test_split to separate training and testing data

6. Evaluation
Generated confusion matrix using sklearn.metrics.confusion_matrix

Displayed it using a heatmap via seaborn

Printed classification report to assess precision, recall, and F1-score

📊 Sample Output
Confusion Matrix
Classification Report
