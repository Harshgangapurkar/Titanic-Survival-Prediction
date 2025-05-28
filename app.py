import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic EDA App", layout="wide")
st.title("🚢 Titanic EDA & Survival Prediction")

# Sidebar for CSV Upload
st.sidebar.header("Upload Titanic CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load dataset
def load_data():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("titanic.csv")  # Your local file
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Data")
genders = df['Sex'].dropna().unique().tolist()
classes = df['Pclass'].dropna().unique().tolist()
selected_gender = st.sidebar.multiselect("Select Gender", genders, default=genders)
selected_class = st.sidebar.multiselect("Select Class", classes, default=classes)

filtered_df = df[df['Sex'].isin(selected_gender) & df['Pclass'].isin(selected_class)]

# Create Tabs
tabs = st.tabs(["📄 Overview", "📊 Visualizations", "🔍 Feature Explorer", "🤖 Survival Predictor"])

# Overview Tab
with tabs[0]:
    st.subheader("First Look at the Data")
    st.dataframe(filtered_df.head())

    st.subheader("Descriptive Statistics")
    st.write(filtered_df.describe())

    st.subheader("Missing Values")
    st.write(filtered_df.isnull().sum())

# Visualization Tab
with tabs[1]:
    st.subheader("Visual Analysis")

    st.write("### Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Survived', ax=ax1)
    ax1.set_xticklabels(['Not Survived', 'Survived'])
    st.pyplot(fig1)

    st.write("### Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df['Age'].dropna(), kde=True, ax=ax2)
    st.pyplot(fig2)

    st.write("### Class vs Survival")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=filtered_df, x='Pclass', hue='Survived', ax=ax3)
    ax3.set_xticklabels(['1st', '2nd', '3rd'])
    st.pyplot(fig3)

# Feature Explorer Tab
with tabs[2]:
    st.subheader("Explore Features")
    num_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    col = st.selectbox("Select Numeric Column", num_cols)

    st.write(f"### Boxplot for {col}")
    fig4, ax4 = plt.subplots()
    sns.boxplot(y=filtered_df[col], ax=ax4)
    st.pyplot(fig4)

    st.write(f"### Summary Statistics for {col}")
    st.write(filtered_df[col].describe())

# Prediction Tab
with tabs[3]:
    st.subheader("Predict Survival")

    model_df = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']].dropna()
    model_df['Sex'] = model_df['Sex'].map({'male': 0, 'female': 1})

    X = model_df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
    y = model_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.markdown("### Try It Out")
    pclass = st.selectbox("Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.radio("Sex", ['male', 'female'])
    age = st.slider("Age", 0, 80, 25)
    fare = st.slider("Fare", 0.0, 500.0, 50.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [0 if sex == 'male' else 1],
        'Age': [age],
        'Fare': [fare],
        'SibSp': [sibsp],
        'Parch': [parch]
    })

    prediction = clf.predict(input_data)[0]
    outcome = "Survived" if prediction == 1 else "Did Not Survive"
    st.success(f"The model predicts: {outcome}")
