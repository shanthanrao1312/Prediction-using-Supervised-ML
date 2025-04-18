🚀 Prediction using Supervised Machine Learning

Welcome to this repository where I explore Supervised Machine Learning to predict student scores based on study hours! This project demonstrates how to build a simple yet effective linear regression model step-by-step using Python. The goal is to predict scores based on the number of hours students study, showcasing the entire process of data exploration, model training, and making predictions.

📊 Overview

In this project, I used a dataset containing two columns:

Hours : Number of study hours.

Scores : Corresponding scores achieved by students.

The project involves the following key steps:

Data Exploration : Understanding the dataset and its structure.

Data Preprocessing : Preparing the data for modeling by splitting it into training and testing sets.

Model Training : Building and training a Linear Regression model.

Visualization : Plotting the regression line to interpret the relationship between study hours and scores.

Prediction : Using the trained model to predict scores for new study hours.

This project serves as a foundational example of how supervised machine learning can be applied to real-world problems.

📂 Dataset

It contains two columns:

Hours : Number of study hours.

Scores : Corresponding scores achieved by students.

This simple yet insightful dataset is ideal for beginners to practice regression techniques.

🔧 Technologies Used

I leveraged the following tools and libraries to complete this project:

Programming Language : Python

Libraries :

pandas: For data manipulation and analysis.

numpy: For numerical computations.

matplotlib: For data visualization.

scikit-learn: For building and evaluating machine learning models.

Environment : Jupyter Notebook for an interactive coding experience.

🛠️ Project Workflow

Here’s a detailed breakdown of how I approached the project:

Step 1: Reading the Dataset

I started by loading the dataset using pandas.read_csv. The dataset contains two columns: Hours and Scores. After loading the data, I extracted the feature (Hours) and target variable (Scores) for further processing.

s_data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

X = s_data.iloc[:, :-1].values  # Feature (Hours)

y = s_data.iloc[:, 1].values    # Target (Scores)

Step 2: Training the Model

Next, I split the dataset into training and testing sets using train_test_split from scikit-learn. I then trained a Linear Regression model using the training data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print("Training complete.")

Step 3: Plotting the Regression Line

After training the model, I plotted the regression line to visualize the relationship between study hours and scores. The regression line was calculated using the slope (m) and intercept (c) obtained from the trained model.

line = regressor.coef_ * X + regressor.intercept_

plt.scatter(X, y)

plt.plot(X, line)

plt.show()

Step 4: Making Predictions

Finally, I used the trained model to predict the score for a student who studies 9.25 hours . 

The prediction was calculated using the equation of the line (y = mx + c).

predicted_score = 9.91 * 9.25 + 2.018  # y = mx + c

print(predicted_score)  # Output: 93.6855

📈 Key Insights and Visualizations

Visualization: Scatter Plot with Regression Line

![image](https://github.com/user-attachments/assets/74c9e032-7719-49e5-b3a5-401c06722b19)


Caption: A scatter plot showing the relationship between study hours and scores, along with the regression line.

Model Parameters

Intercept (c)

2.01816

Slope (m)

9.91065648

🌟 Why This Project?

This project showcases the simplicity and power of supervised machine learning. By focusing on a single feature (Hours) and a target variable (Scores), I demonstrated how to:

Prepare and preprocess data for modeling.

Train a Linear Regression model.

Visualize the relationship between features and targets.

Make predictions using the trained model.

While this project does not include advanced evaluation metrics like MAE, MSE, or R-squared, it provides a clear and beginner-friendly introduction to supervised machine learning.

Thank you for visiting this repository!
