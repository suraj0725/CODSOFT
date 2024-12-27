TITANIC-SURVIVAL-PREDICTION-CodeSoft
Titanic Survival Prediction This repository contains Python code for predicting survival on the Titanic using machine learning models. The code leverages popular libraries like Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. It preprocesses the Titanic dataset, trains and tunes Random Forest and Gradient Boosting classifiers, and combines them into a Voting Classifier ensemble for final predictions. The repository also includes data visualization for better understanding the dataset.

Introduction

The Titanic Survival Prediction project aims to build a predictive model to determine whether a passenger survived or not based on various features. It employs two popular machine learning algorithms, Random Forest and Gradient Boosting, and combines their predictions using a Voting Classifier ensemble for better accuracy.

Dependencies

Make sure you have the following Python libraries installed:

Pandas NumPy Matplotlib Seaborn scikit-learn You can install them using pip:

Copy code pip install pandas numpy matplotlib seaborn scikit-learn

Dataset

The dataset used in this project is stored in the 'tested.csv' file. It contains information about Titanic passengers, including features like age, gender, class, and more.

Data Preprocessing

Data preprocessing involves handling missing values, transforming categorical variables, and splitting the dataset into training and testing sets.

Missing values in 'Age', 'Embarked', and 'Fare' are filled with appropriate values. 'FamilySize' and 'Title' columns are created from existing features. 'Title' categories are grouped for better model performance. Unnecessary columns ('PassengerId', 'Name', 'Ticket', 'Cabin') are dropped. Categorical variables are one-hot encoded.

Feature Engineering

Feature selection and scaling are essential steps in machine learning.

Feature selection is performed using the SelectKBest method with an Anova F-value scoring function. Features are scaled using StandardScaler to ensure that all features have the same scale.

Model Training

Two classifiers, Random Forest and Gradient Boosting, are trained on the preprocessed and scaled data. Hyperparameter tuning is performed using GridSearchCV to find the best model parameters.

Model Evaluation

The performance of the ensemble model is evaluated using cross-validation and metrics like accuracy, classification report, and confusion matrix.

Data Visualization

Data visualization is essential for understanding the dataset and model results.

A correlation matrix heatmap shows the relationships between features. An age distribution plot by survival status provides insights into passenger age. A survival rate by family size bar plot highlights the influence of family size on survival. A fare distribution by class and survival box plot helps analyze fare differences between passenger classes. A pair plot of features visualizes relationships between features and their impact on survival. Feel free to explore the code and adapt it to your needs for further analysis or modeling tasks. Happy coding!
