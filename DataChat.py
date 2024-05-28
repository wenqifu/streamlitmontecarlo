import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}
API_URL = "https://api.openai.com/v1/chat/completions"

def ask_gpt3(prompt):
    """ Function to query the OpenAI API with a prompt """
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        error_info = response.json()
        return f"Error communicating with AI service: Status code {response.status_code}, {error_info}"

# File upload widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Generate descriptive statistics and data types
    descriptive_prompt = "Descriptive Analysis: "
    if any(data.dtypes.apply(lambda x: x in ['float64', 'int64', 'object'])):
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        object_data = data.select_dtypes(include=['object'])

        numeric_descriptives = {
            'Type': 'Numerical',
            'Data Types': numeric_data.dtypes.apply(lambda x: str(x)).to_dict(),
            'Mean': numeric_data.mean().to_dict(),
            'Variance': numeric_data.var().to_dict(),
            'Standard Deviation': numeric_data.std().to_dict(),
            'Range': (numeric_data.max() - numeric_data.min()).to_dict(),
            'Minimum': numeric_data.min().to_dict(),
            'Maximum': numeric_data.max().to_dict()
        }

        category_descriptives = {
            'Type': 'Categorical',
            'Data Types': object_data.dtypes.apply(lambda x: str(x)).to_dict(),
            'Counts': object_data.apply(lambda x: x.value_counts().to_dict()).to_dict()
        }

        descriptive_prompt += f"Numerical: {numeric_descriptives}, Categorical: {category_descriptives}."

    # Allow the user to ask questions about the dataset
    user_input = st.text_input("Ask GPT-3 any question about the dataset:")
    if user_input:
        full_prompt = descriptive_prompt + " " + user_input
        ai_response = ask_gpt3(full_prompt)
        st.text_area("GPT-3 Response:", value=ai_response, height=200)



    # Column selection for visualization
    selected_columns = st.multiselect('Select 2 or 3 columns to visualize', data.columns)
    plot_types = {('N', 'N'): 'Scatter Plot', ('N', 'C'): 'Box Plot', ('C', 'N'): 'Box Plot',
                  ('C', 'C'): 'Bar Plot', ('N', 'N', 'N'): '3D Scatter Plot', ('N', 'N', 'C'): 'Grouped Scatter Plot',
                  ('N', 'C', 'C'): 'Grouped Box Plot', ('C', 'C', 'C'): 'Stacked Bar Plot'}
    if len(selected_columns) == 2 or len(selected_columns) == 3:
        types = tuple('N' if pd.api.types.is_numeric_dtype(data[col]) else 'C' for col in selected_columns)
        selected_plot = plot_types.get(types)
        if selected_plot:
            st.write(f"Selected plot type: {selected_plot}")
            # Implement plotting logic based on the selected_plot type here


    # Normality test for a selected numerical column
    if any(data.dtypes.apply(lambda x: x in ['float64', 'int64'])):
        st.write("### Normality Test for Numerical Data")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        column_to_test = st.selectbox("Select a Numerical Column to Test for Normality", numeric_data.columns)
        if st.button('Test Normality'):
            # Perform Shapiro-Wilk test
            test_statistic, p_value = shapiro(data[column_to_test].dropna())  # dropna to handle NaN
            shapiro_result = f"Test Statistic: {test_statistic}, P-value: {p_value}"
            st.write(shapiro_result)

            # Ask GPT-3.5 to decide based on the Shapiro-Wilk test
            normality_decision = ask_gpt3(f"The Shapiro-Wilk test resulted in a p-value of {p_value}. Explain simply if the data is normally distributed.")
            st.write(normality_decision)

            # Generate and display a QQ plot
            fig, ax = plt.subplots()
            qqplot(data[column_to_test].dropna(), line ='45', ax=ax)
            plt.title("QQ Plot")
            st.pyplot(fig)

            # Ask GPT-3.5 to interpret the QQ plot
            qq_plot_description = "A QQ plot showing the quantiles of the data versus the quantiles of a normal distribution."
            qq_plot_judgment = ask_gpt3(f"Review this QQ plot description and simply write a decision if the data seems normal: {qq_plot_description}")
            st.write("GPT-3.5's judgment on the QQ plot:")
            st.write(qq_plot_judgment)
