import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load custom CSS
local_css("style.css")


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv(r'income_evaluation.csv')
    df.columns = ['age', 'workclass', 'final_weight', 'education', 'education_num',
                  'martial_status', 'occupation', 'relationship', 'race', 'sex',
                  'capital_gain', 'capital_loss', 'hrs_per_week', 'native_country', 'income']
    return df


@st.cache_data
def preprocess_data(df):
    # Convert income to binary
    df['income'] = [1 if value == ' >50K' else 0 for value in df['income'].values]

    # Handle missing values
    df['workclass'] = np.where(df.workclass == ' ?', np.nan, df['workclass'])
    df['native_country'] = np.where(df.native_country == ' ?', np.nan, df['native_country'])
    df['occupation'] = np.where(df.occupation == ' ?', np.nan, df['occupation'])
    df.dropna(axis=0, inplace=True)

    # Label encoding
    categorical_cols = ['workclass', 'education', 'martial_status', 'occupation',
                        'relationship', 'race', 'native_country']

    for col in categorical_cols:
        labels = {v: k for k, v in enumerate(df[col].unique())}
        df[col] = df[col].map(labels)

    # Convert sex to binary
    df['sex'] = np.where(df.sex == ' Male', 1, 0)

    return df


# Load and process data
df = load_data()
df = preprocess_data(df)

# Check if model exists, if not train and save it
model_path = 'income_model.pkl'
if not os.path.exists(model_path):
    with st.spinner("Training model... Please wait..."):
        # Prepare data for training
        X = df.drop('income', axis=1).values
        y = df['income'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump((model, sc), f)
else:
    # Load the trained model and scaler
    with open(model_path, 'rb') as f:
        model, sc = pickle.load(f)

# Create mappings for dropdown options
workclass_mapping = {k: v for v, k in enumerate(df['workclass'].unique())}
education_mapping = {k: v for v, k in enumerate(df['education'].unique())}
martial_status_mapping = {k: v for v, k in enumerate(df['martial_status'].unique())}
occupation_mapping = {k: v for v, k in enumerate(df['occupation'].unique())}
relationship_mapping = {k: v for v, k in enumerate(df['relationship'].unique())}
race_mapping = {k: v for v, k in enumerate(df['race'].unique())}
native_country_mapping = {k: v for v, k in enumerate(df['native_country'].unique())}

# Reverse mappings for display
workclass_options = {v: k for k, v in workclass_mapping.items()}
education_options = {v: k for k, v in education_mapping.items()}
martial_status_options = {v: k for k, v in martial_status_mapping.items()}
occupation_options = {v: k for k, v in occupation_mapping.items()}
relationship_options = {v: k for k, v in relationship_mapping.items()}
race_options = {v: k for k, v in race_mapping.items()}
native_country_options = {v: k for k, v in native_country_mapping.items()}

# App Header
st.markdown("""
    <div class='header'>
        <h1>💰 Income Classifier</h1>
        <p>Predict whether income exceeds $50K/year based on census data</p>
    </div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio("Menu", ["🏠 Home", "🔮 Predict", "📊 Analysis", "📈 Model Metrics"])

if page == "🏠 Home":
    st.markdown("""
    <div class='home-container'>
        <h2>Welcome to Income Classifier</h2>
        <p>An interactive application that predicts income levels using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Key Features
        - Income prediction with 85%+ accuracy
        - Interactive data visualizations
        - Model performance metrics
        - User-friendly interface

        ### How It Works
        1. Navigate to the Predict page
        2. Fill in the demographic information
        3. Get instant prediction with confidence score
        """)

    with col2:
        st.image(
            "https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80",
            caption="Data Science in Action")

elif page == "🔮 Predict":
    st.header("Income Prediction")
    st.markdown("Fill in the details to predict income level")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Details")
            age = st.slider("Age", 17, 90, 35)
            sex = st.radio("Gender", ["Male", "Female"])
            race = st.selectbox("Race", options=list(race_options.keys()))
            relationship = st.selectbox("Relationship", options=list(relationship_options.keys()))
            martial_status = st.selectbox("Marital Status", options=list(martial_status_options.keys()))

        with col2:
            st.subheader("Employment Details")
            workclass = st.selectbox("Work Class", options=list(workclass_options.keys()))
            occupation = st.selectbox("Occupation", options=list(occupation_options.keys()))
            education = st.selectbox("Education", options=list(education_options.keys()))
            education_num = st.slider("Education Years", 1, 16, 9)
            hrs_per_week = st.slider("Hours Per Week", 1, 100, 40)
            native_country = st.selectbox("Native Country", options=list(native_country_options.keys()))

        st.subheader("Financial Details")
        col3, col4, col5 = st.columns(3)
        with col3:
            final_weight = st.number_input("Final Weight", min_value=0, value=100000)
        with col4:
            capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        with col5:
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

        submitted = st.form_submit_button("Predict Income", use_container_width=True)

        if submitted:
            with st.spinner("Analyzing data..."):
                try:
                    # Prepare input data
                    form_data = {
                        'age': age,
                        'workclass': workclass_mapping[workclass],
                        'final_weight': final_weight,
                        'education': education_mapping[education],
                        'education_num': education_num,
                        'martial_status': martial_status_mapping[martial_status],
                        'occupation': occupation_mapping[occupation],
                        'relationship': relationship_mapping[relationship],
                        'race': race_mapping[race],
                        'sex': 1 if sex == "Male" else 0,
                        'capital_gain': capital_gain,
                        'capital_loss': capital_loss,
                        'hrs_per_week': hrs_per_week,
                        'native_country': native_country_mapping[native_country]
                    }

                    # Create feature array
                    features = np.array([list(form_data.values())])

                    # Scale features using the saved scaler
                    features_scaled = sc.transform(features)

                    # Make prediction
                    prediction = model.predict(features_scaled)
                    probability = model.predict_proba(features_scaled)[0][1]

                    result = "> $50K/year" if prediction[0] == 1 else "≤ $50K/year"
                    confidence = round(probability * 100, 2)

                    # Display result with style
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class='prediction-high'>
                            <h2>Prediction: {result}</h2>
                            <p>Confidence: {confidence}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='prediction-low'>
                            <h2>Prediction: {result}</h2>
                            <p>Confidence: {confidence}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "📊 Analysis":
    st.header("Data Analysis Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Income Distribution",
        "👵 Age Analysis",
        "🎓 Education Impact",
        "🔗 Correlation"
    ])

    with tab1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='income', data=df, palette="viridis", ax=ax)
        ax.set_xticklabels(["≤ $50K", "> $50K"])
        ax.set_title("Distribution of Income Levels")
        st.pyplot(fig)

    with tab2:
        st.subheader("Age vs Income")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='income', y='age', data=df, palette="coolwarm", ax=ax)
        ax.set_xticklabels(["≤ $50K", "> $50K"])
        ax.set_title("Age Distribution by Income Level")
        st.pyplot(fig)

    with tab3:
        st.subheader("Education Level vs Income")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x='education', hue='income', data=df, palette="Set2", ax=ax)
        plt.xticks(rotation=45)
        ax.set_title("Education Impact on Income")
        ax.legend(title="Income", labels=["≤ $50K", "> $50K"])
        st.pyplot(fig)

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

elif page == "📈 Model Metrics":
    st.header("Model Performance Metrics")

    # Prepare data for metrics
    X = df.drop('income', axis=1).values
    y = df['income'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.metric("Accuracy Score", f"{accuracy_score(y_test, pred) * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>Developed by Saran Prakash B</p>
    <p>© 2025 Income Classifier</p>
</div>
""", unsafe_allow_html=True)