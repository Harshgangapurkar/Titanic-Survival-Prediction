🚢 Titanic Survival Prediction using Logistic Regression
This project leverages supervised machine learning to predict the survival of passengers aboard the Titanic. Using logistic regression, we explore the relationships between key features like age, gender, and passenger class, and apply classification techniques to determine survival outcomes.

📂 Dataset
Source: titanic.csv

Contains demographic and travel details of 891 passengers including:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, and Survived (target)

🧠 Project Overview
🔧 1. Initialization
Imported key libraries: pandas, numpy, matplotlib, seaborn, and sklearn.

📊 2. Exploratory Data Analysis (EDA)
Explored dataset structure using .head(), .info(), and .describe()

Visualized survival distribution across features such as:

Gender (Sex)

Passenger class (Pclass)

Age groups

Identified correlations and patterns using seaborn plots.

🧹 3. Data Cleaning & Feature Engineering
Handled missing values in Age and Embarked

Converted categorical features into numeric form using label encoding

Dropped irrelevant features like Cabin (due to high missingness)

🧪 4. Model Training
Split the dataset using train_test_split

Trained a Logistic Regression model using sklearn.linear_model.LogisticRegression

Performed predictions on the test set

📈 5. Evaluation
Assessed performance using:

Confusion Matrix (visualized via heatmap)

Classification Report (Precision, Recall, F1-score)

Observed areas of model strength and weakness

🔍 Key Insights
Females had significantly higher survival rates than males

1st class passengers had better survival odds

Age and family size impacted survival probabilities

🚀 Getting Started
Clone the repository

Place titanic.csv in the root directory

Launch the notebook and run all cells

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
📌 Future Work
Feature scaling and normalization

Hyperparameter tuning

Cross-validation

Comparison with other models like Random Forest, KNN, and SVM

🙏 Acknowledgements
Dataset: Kaggle Titanic: Machine Learning from Disaster

Model: Logistic Regression using scikit-learn

📁 Output Preview
🎯 Confusion Matrix

📋 Classification Report

📊 Visual Explorations (Bar plots, heatmaps)
