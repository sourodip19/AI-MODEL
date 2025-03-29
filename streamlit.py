import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, mean_squared_error, r2_score)

def main():
    st.set_page_config(
        page_title="ML Visualization Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Machine Learning Visualization Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # File upload section
    with st.sidebar:
        st.header("1. Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Main analysis section
    if st.session_state.df is not None:
        st.header("2. Data Analysis")
        
        # Show basic data info
        with st.expander("Dataset Preview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("First 5 rows:")
                st.dataframe(st.session_state.df.head())
            with col2:
                st.write("Dataset Summary:")
                st.write(st.session_state.df.describe())

        # Analysis type selection
        st.subheader("Analysis Type")
        analysis_type = st.radio(
            "Select analysis type:",
            ["Classification", "Regression"],
            horizontal=True
        )

        # Perform analysis
        if st.button("Run Analysis"):
            with st.spinner("Analyzing data..."):
                try:
                    X = st.session_state.df.iloc[:, :-1]
                    y = st.session_state.df.iloc[:, -1]

                    if analysis_type == "Classification":
                        perform_classification(X, y)
                    else:
                        perform_regression(X, y)
                    
                    st.session_state.analysis_done = True
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

        # Show visualizations if analysis done
        if st.session_state.analysis_done:
            st.header("3. Visualizations")
            show_visualizations()

        # Always show correlation matrix
        st.header("Feature Correlations")
        show_correlation_matrix()

def perform_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.metric("Accuracy Score", f"{accuracy_score(y_test, y_pred):.2%}")

def perform_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regression Metrics")
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
        
        st.write("Coefficients:")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })
        st.dataframe(coef_df)
    
    with col2:
        st.subheader("Prediction Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)

def show_visualizations():
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Pie Chart", "Bar Chart", "Scatter Plot", "Histogram", "KDE Plot"]
    )
    
    try:
        if viz_type == "Pie Chart":
            fig, ax = plt.subplots()
            st.session_state.df.iloc[:, -1].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Bar Chart":
            fig, ax = plt.subplots()
            st.session_state.df.iloc[:, -1].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Scatter Plot":
            if len(st.session_state.df.columns) < 3:
                st.warning("Need at least 2 features for scatter plot")
                return
            fig, ax = plt.subplots()
            ax.scatter(st.session_state.df.iloc[:, 0], 
                       st.session_state.df.iloc[:, 1], 
                       c=st.session_state.df.iloc[:, -1])
            st.pyplot(fig)
            
        elif viz_type in ["Histogram", "KDE Plot"]:
            col = st.selectbox("Select column", st.session_state.df.columns)
            fig, ax = plt.subplots()
            if viz_type == "Histogram":
                sns.histplot(st.session_state.df[col], kde=True, ax=ax)
            else:
                sns.kdeplot(st.session_state.df[col], ax=ax)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def show_correlation_matrix():
    numeric_df = st.session_state.df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation matrix")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()