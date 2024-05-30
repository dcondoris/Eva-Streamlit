import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Define custom CSS for light business blue tone and white tone
st.markdown("""
    <style>
        .main {
            background-color: #ffffff; /* White tone for main page */
        }
        .sidebar .sidebar-content {
            background-color: #87CEFA; /* Light sky blue tone for sidebar */
        }
        .reportview-container .main .block-container {
            background-color: #ffffff; /* White tone for main content area */
        }
        .css-18e3th9, .css-1d391kg {
            background-color: #87CEFA; /* Light sky blue tone for upper menu */
        }
        .stMarkdown p {
            color: #495057;
        }
        .stButton>button {
            background-color: #007bff;
            color: #fff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #343a40;
        }
        .participants-title {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Menu")
menu_items = ["Project Overview", "Datasets", "Exploratory Analysis", "Methodology", "Modeling", "Demo"]
selected_menu = st.sidebar.radio("", menu_items, index=0, key="menu")

# Load the datasets
@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv('X_train_update.csv')
        Y_train = pd.read_csv('Y_train_CVwO8PX.csv')
        X_test = pd.read_csv('X_test_update.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None, None

    return X_train, Y_train, X_test

@st.cache_data
def load_images(image_folder, num_images=5):
    images = []
    for folder in ['image_train', 'image_test']:
        folder_path = os.path.join(image_folder, folder)
        images.extend([os.path.join(folder_path, img) for img in os.listdir(folder_path)[:num_images]])
    return images

X_train, Y_train, X_test = load_data()

if X_train is not None and Y_train is not None and X_test is not None:
    # Define a function to display sample images
    def display_images(image_paths):
        fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
        for img_path, ax in zip(image_paths, axes):
            image = Image.open(img_path)
            ax.imshow(image)
            ax.axis('off')
        st.pyplot(fig)

    # Function to analyze datasets
    def analyze_dataset(df, name):
        st.subheader(f"Analysis of {name}")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Number of duplicates:** {df.duplicated().sum()}")
        st.write(f"**Number of null values:**\n{df.isnull().sum()}")
        if 'designation' in df.columns:
            st.write(f"**Number of missing designations:** {df['designation'].isnull().sum()}")
        if 'description' in df.columns:
            st.write(f"**Number of missing descriptions:** {df['description'].isnull().sum()}")
        if 'designation' in df.columns:
            st.write(
                f"**Number of words in titles:**\n{df['designation'].apply(lambda x: len(str(x).split())).describe()}")
        if 'description' in df.columns:
            st.write(
                f"**Number of words in descriptions:**\n{df['description'].apply(lambda x: len(str(x).split())).describe()}")
        if 'imageid' in df.columns:
            st.write(f"**Repetition of images:**\n{df['imageid'].value_counts().head()}")
        st.write("---")

    # Main Page - Project Overview
    if selected_menu == "Project Overview":
        st.title("Rakuten Product Classification")
        st.header("Context")
        st.markdown("""
        Accurate product classification enhances e-commerce efficiency by improving search accuracy, personalized recommendations, and inventory management, leading to better customer satisfaction and increased sales. For Rakuten, this means reduced operational costs and higher revenue.

        This project involves large-scale multimodal classification, combining text mining and image processing to categorize products into predefined types. It requires proficiency in machine learning, deep learning, and data mining, aligning well with our team's expertise and the skills acquired during our training.
        """)

        st.header("Project Objectives")
        st.markdown("""
        To summarize, our objectives for this project were:
        1. Train various models to classify products into different categories.
        2. Fine-tune a method leveraging the most performant models.
        3. Apply the developed method on the test set provided by Rakuten.
        4. Develop an application to classify any new product in real-time.
        """)

    # Datasets
    if selected_menu == "Datasets":
        st.title("Datasets")

        st.header("Data Input Summary")
        st.markdown("""
        The dataset includes approximately 99,000 product listings, with around 84,916 entries in the training set and 13,812 entries in the test set. The image data is around 2.2 GB, containing images of size 500x500 pixels. in the test set. Each image is 500x500 pixels in size.

        The data are freely available on the Kaggle website.
        """)

        # Display the uploaded images instead of the text description
        #st.header("Data Input Summary - Visual Representation")
        image_path_1 = 'input_dataset.png'  # Path to the first image
        st.image(image_path_1, caption='Dataset Structure 1', use_column_width=True)

        # Display the head of each dataset
        st.header("Dataset Previews")
        st.subheader("X_train.csv")
        st.write(X_train.head())

        st.subheader("Y_train.csv")
        st.write(Y_train.head())

        st.subheader("X_test.csv")
        st.write(X_test.head())

        # Display sample images
        st.header("Sample Images")
        image_paths = load_images('images', num_images=5)
        display_images(image_paths)

        # Analyze each dataset
        st.header("Detailed Analysis of Datasets")
        analyze_dataset(X_train, "X_train.csv")
        analyze_dataset(Y_train, "Y_train.csv")
        analyze_dataset(X_test, "X_test.csv")

        # Additional plots
        st.header("Data Distribution and Visualization")

        # Plot distribution of product type codes in Y_train
        st.subheader("Distribution of Product Type Codes in Y_train")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(y=Y_train.iloc[:, 0], ax=ax1)
        ax1.set_title('Distribution of Product Type Codes')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Product Type Code')
        st.pyplot(fig1)


        # Plot length of descriptions in X_train
        st.subheader("Length of Descriptions in X_train")
        X_train['Description Length'] = X_train['description'].apply(lambda x: len(str(x).split()))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(X_train['Description Length'], bins=50, kde=True, ax=ax2)
        ax2.set_title('Distribution of Description Lengths')
        ax2.set_xlabel('Description Length')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)


# Placeholder for other sections
if selected_menu == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.markdown("Details about the exploratory analysis go here.")

if selected_menu == "Methodology":
    st.title("Methodology")
    st.markdown("Details about the methodology go here.")

if selected_menu == "Modeling":
    st.title("Modeling")
    st.markdown("Details about the modeling go here.")

if selected_menu == "Demo":
    st.title("Demo")
    st.markdown("Details about the demo go here.")

# Project team in the sidebar
st.sidebar.markdown("**Data Science Project**")
st.sidebar.markdown("**Bootcamp March 2024**")
st.sidebar.markdown('<p class="participants-title">Participants</p>', unsafe_allow_html=True)

team_members = [
    {"name": "João Pedro Kerr Catunda", "linkedin": None},
    {"name": "Mani Chandan Naru", "linkedin": "https://www.linkedin.com/in/mani-cn/"},
    {"name": "Eva Losada Barreiro", "linkedin": "https://www.linkedin.com/in/evalosadabarreiro/?locale=de_DE"}
]

for member in team_members:
    st.sidebar.markdown(f"**{member['name']}**")
    if member['linkedin']:
        st.sidebar.markdown(f"[LinkedIn]({member['linkedin']})")
