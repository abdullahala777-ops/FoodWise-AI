# FoodWise AI – Food Waste Prediction

## Project Overview
FoodWise AI is a machine learning project designed to predict food waste using historical data. The project aims to support better food inventory management and reduce food waste.

## Project Objectives
- Analyze food waste data using Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Train and evaluate multiple machine learning regression models.
- Compare model performance.
- Provide an interactive prediction application using Streamlit.

## Machine Learning Models
The project includes the following models:
1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Baseline Mean Model

## Project Structure
```text
food/
├── Data/
│   ├── Original/
│   └── Cleaned/
├── Notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb
├── Models/
├── Presentation/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Joblib

## How to Run the Application

1. Clone the repository:
```bash
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Application Features
- Enter input data to predict food waste.
- Select a machine learning model.
- Compare predictions from different models.
- View prediction results through an interactive interface.

## Project Status
The project includes data analysis, preprocessing, model training, evaluation, and an interactive Streamlit application.
