import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as UMAP



# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar input for user text
user_input = st.sidebar.text_input('Enter something')

# Sidebar file uploader for CSV file
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Function to load data from the uploaded CSV file
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        st.session_state.data = df

# Check if a file is uploaded and data is not already loaded
if uploaded_file is not None and st.session_state.data is None:
    load_data(uploaded_file)

# Display loaded data or message if no data is loaded
if st.session_state.data is not None:
    st.write("Data loaded:")
    st.dataframe(st.session_state.data.head())
else:
    st.write("No data loaded yet.")

# Process user input if provided
if user_input:
    st.write("User input:", user_input)

if st.session_state.data is not None:
    data = st.session_state.data
    backup_data = data.copy()

    # Function to determine plot type based on selected columns
    def determine_plot_type(columns):
        types = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                types.append('Numerical')
            elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                types.append('Categorical')
            else:
                types.append('Other')  # For any type that is not clearly numerical or categorical

        type_string = ''.join([t[0] for t in types])  # 'NN', 'NC', 'CC', 'NNN', 'NNC', 'NCC', 'CCC'

        plot_map = {
            'NN': 'Scatter Plot',
            'NC': 'Box Plot',
            'CN': 'Box Plot',
            'CC': 'Bar Plot',
            'NNN': '3D Scatter Plot',
            'NNC': 'Grouped Scatter Plot',
            'NCC': 'Grouped Box Plot',
            'CCC': 'Stacked Bar Plot'
        }

        return plot_map.get(type_string, None)

    # Implementing plotting functions
    def plot_scatter(data, selected_columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]])
        plt.title(f'Scatter Plot of {selected_columns[0]} vs {selected_columns[1]}')
        st.pyplot(plt)

    def plot_box(data, selected_columns):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[selected_columns[0]], y=data[selected_columns[1]])
        plt.title(f'Box Plot of {selected_columns[1]} by {selected_columns[0]}')
        st.pyplot(plt)

    def plot_bar(data, selected_columns):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data[selected_columns[0]], hue=data[selected_columns[1]])
        plt.title(f'Bar Plot of {selected_columns[0]} grouped by {selected_columns[1]}')
        st.pyplot(plt)

    def plot_3d_scatter(data, selected_columns):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[selected_columns[0]], data[selected_columns[1]], data[selected_columns[2]])
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        ax.set_zlabel(selected_columns[2])
        plt.title('3D Scatter Plot')
        st.pyplot(fig)

    def plot_grouped_scatter(data, selected_columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], hue=data[selected_columns[2]])
        plt.title(f'Grouped Scatter Plot of {selected_columns[0]} vs {selected_columns[1]}')
        st.pyplot(plt)

    def plot_grouped_box(data, selected_columns):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[selected_columns[1]], y=data[selected_columns[0]], hue=data[selected_columns[2]])
        plt.title(f'Grouped Box Plot of {selected_columns[0]} by {selected_columns[1]} and {selected_columns[2]}')
        st.pyplot(plt)

    def plot_stacked_bar(data, selected_columns):
        ct_before = pd.crosstab(index=data[selected_columns[0]], columns=[data[selected_columns[1]], data[selected_columns[2]]])
        ct_before.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Stacked Bar Plot')
        st.pyplot(plt)

    plot_functions = {
        'Scatter Plot': plot_scatter,
        'Box Plot': plot_box,
        'Bar Plot': plot_bar,
        '3D Scatter Plot': plot_3d_scatter,
        'Grouped Scatter Plot': plot_grouped_scatter,
        'Grouped Box Plot': plot_grouped_box,
        'Stacked Bar Plot': plot_stacked_bar
    }

    # Function to handle scaling
    def scale_data(method, columns):
        global backup_data, data
        if backup_data is None:
            backup_data = data.copy()

        scaler = None
        if method == 'min-max':
            scaler = MinMaxScaler()
        elif method == 'z-score':
            scaler = StandardScaler()
        elif method == 'l1':
            scaler = Normalizer(norm='l1')
        elif method == 'l2':
            scaler = Normalizer(norm='l2')

        for column in columns:
            if column in data.columns:
                data[[column]] = scaler.fit_transform(data[[column]])
                st.write(f"Column {column} is scaled using {method} method.")
                st.write(data.head())
            else:
                st.write(f"Column {column} does not exist in the dataset.")

    # Function to revert scaling
    def revert_scaling():
        global backup_data, data
        if backup_data is not None:
            st.write("Reverting to original data:")
            st.write(backup_data.head())
            data = backup_data
            backup_data = None
            return data
        else:
            st.write("No scaling to revert.")
            return data

    st.write("### Enter your command:")
    command = st.text_input("Type your command here (e.g., 'column1, column2' or 'scale, column1, column2' or 'revert')")


    # Function to apply transformations
    def transform_data(method, columns):
        global backup_data, data
        if backup_data is None:
            backup_data = data.copy()

        transformer = None
        if method == 'log':
            for column in columns:
                if column in data.columns:
                    data[[column]] = np.log1p(data[[column]])  # Using log1p to handle log(0)
                    st.write(f"Column {column} is transformed using {method} method.")
                    st.write(data.head())
                else:
                    st.write(f"Column {column} does not exist in the dataset.")
        elif method == 'box-cox':
            transformer = PowerTransformer(method='box-cox', standardize=False)
            for column in columns:
                if column in data.columns:
                    if (data[column] > 0).all():  # Box-Cox requires all positive values
                        data[[column]] = transformer.fit_transform(data[[column]])
                        st.write(f"Column {column} is transformed using {method} method.")
                        st.write(data.head())
                    else:
                        st.write(f"Column {column} contains non-positive values and cannot be transformed using {method} method.")
                else:
                    st.write(f"Column {column} does not exist in the dataset.")


    def dim_reduction(method, columns):
        global data
        if all(col in data.columns for col in columns):
            X = data[columns].values
            reducer = None
            if method == 'pca':
                reducer = PCA(n_components=2)
            elif method == 'lda':
                if 'target' in data.columns:
                    y = data['target']
                    reducer = LDA(n_components=2)
                    X_reduced = reducer.fit_transform(X, y)
                else:
                    st.write("LDA requires a 'target' column in the dataset.")
                    return
            elif method == 'tsne':
                reducer = TSNE(n_components=2)
            elif method == 'umap':
                reducer = UMAP(n_components=2)

            if reducer is not None:
                if method != 'lda':
                    X_reduced = reducer.fit_transform(X)
                reduced_df = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
                st.write(f"Dimensionality reduction using {method} method completed.")
                st.write(reduced_df.head())

                # Add the reduced components back to the original DataFrame
                data['Component 1'] = reduced_df['Component 1']
                data['Component 2'] = reduced_df['Component 2']
                data = data.drop(columns=columns)

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='Component 1', y='Component 2', data=data)
                plt.title(f'Dimensionality Reduction using {method.upper()}')
                st.pyplot(plt)
            else:
                st.write(f"Invalid method: {method}")
        else:
            st.write("Some of the specified columns do not exist in the dataset. Please check the column names.")


    # Command processing
    if command:
        actions = command.split(';')
        for action in actions:
            action_parts = action.strip().split(",")
            action_parts = [part.strip() for part in action_parts]

            if action_parts[0].lower() == 'scale':
                if len(action_parts) >= 3:
                    method = action_parts[1].lower()
                    columns = action_parts[2:]
                    try:
                        scale_data(method, columns)
                    except Exception as e:
                        st.write(f"Error scaling columns {columns} using {method} method: {e}")
                else:
                    st.write("Please provide scaling method and columns to scale.")
            elif action_parts[0].lower() == 'transform':
                if len(action_parts) >= 3:
                    method = action_parts[1].lower()
                    columns = action_parts[2:]
                    try:
                        transform_data(method, columns)
                    except Exception as e:
                        st.write(f"Error transforming columns {columns} using {method} method: {e}")
                else:
                    st.write("Please provide transformation method and columns to transform.")
            elif action_parts[0].lower() == 'dim_reduction':
                if len(action_parts) >= 3:
                    method = action_parts[1].lower()
                    columns = action_parts[2:]
                    try:
                        dim_reduction(method, columns)
                    except Exception as e:
                        st.write(f"Error performing dimensionality reduction on columns {columns} using {method} method: {e}")
                else:
                    st.write("Please provide dimensionality reduction method and columns to reduce.")
            elif action_parts[0].lower() == 'revert':
                data = revert_scaling()
            else:
                selected_columns = action_parts
                st.write(f"Trying to plot columns: {selected_columns}")

                if all(col in data.columns for col in selected_columns):
                    if len(selected_columns) == 2 or len(selected_columns) == 3:
                        selected_plot = determine_plot_type(selected_columns)
                        if selected_plot:
                            plot_functions[selected_plot](data, selected_columns)
                        else:
                            st.write("No suitable plot type found for the selected columns.")
                    else:
                        st.write("Please select exactly 2 or 3 columns.")
                else:
                    st.write("Some of the specified columns do not exist in the dataset. Please check the column names.")
