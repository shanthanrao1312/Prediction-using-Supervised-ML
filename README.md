# 🚀 Prediction Using Supervised Machine Learning

Welcome to this repository! Here, I explore **Supervised Machine Learning** to predict student scores based on study hours. This project demonstrates the step-by-step process of building a simple yet effective **Linear Regression** model using Python. The goal is to predict scores based on the number of hours students study, showcasing the entire pipeline from data exploration to prediction.

---

## 📊 Overview

This project revolves around a dataset containing two columns:

- **Hours**: The number of study hours.
- **Scores**: The corresponding scores achieved by students.

The key steps in the project include:

1. **Data Exploration**: Understanding the dataset and its structure.
2. **Data Preprocessing**: Splitting the data into training and testing sets.
3. **Model Training**: Building and training a Linear Regression model.
4. **Visualization**: Plotting the regression line to interpret the relationship between study hours and scores.
5. **Prediction**: Using the trained model to predict scores for new study hours.

By applying these steps, this project serves as a foundational example of how supervised machine learning can be applied to solve real-world problems.

---

## 📂 Dataset

The dataset used in this project is simple yet insightful, making it ideal for beginners to practice regression techniques. It contains the following columns:

- **Hours**: Number of study hours.
- **Scores**: Scores achieved by the students.

You can access the dataset [here](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv).

---

## 🔧 Technologies Used

The following tools and libraries were used to complete this project:

- **Programming Language**: Python
- **Libraries**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib`: For data visualization.
  - `scikit-learn`: For building and evaluating machine learning models.
- **Environment**: Jupyter Notebook for an interactive coding experience.

---

## 🛠️ Project Workflow

Here’s a detailed breakdown of the steps followed in this project:

### Step 1: Reading the Dataset

The dataset was loaded using `pandas.read_csv`. It contains two columns: **Hours** and **Scores**. After loading the data, I extracted the feature (**Hours**) and target variable (**Scores**) for further processing.

```python
s_data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

X = s_data.iloc[:, :-1].values  # Feature (Hours)
y = s_data.iloc[:, 1].values    # Target (Scores)
```

---

### Step 2: Training the Model

I split the dataset into training and testing sets using `train_test_split` from `scikit-learn`. Then, I trained a Linear Regression model on the training data.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")
```

---

### Step 3: Plotting the Regression Line

After training the model, I visualized the relationship between study hours and scores by plotting a scatter plot with the regression line.

```python
import matplotlib.pyplot as plt

line = regressor.coef_ * X + regressor.intercept_

plt.scatter(X, y, color="blue")
plt.plot(X, line, color="red")
plt.title("Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()
```

---

### Step 4: Making Predictions

Finally, I used the trained model to predict the score for a student who studies **9.25 hours**.

```python
predicted_score = regressor.predict([[9.25]])
print(f"Predicted score for 9.25 study hours: {predicted_score[0]:.2f}")
```

---

## 📈 Key Insights and Visualizations

### Visualization: Scatter Plot with Regression Line

The scatter plot below shows the relationship between study hours and scores, along with the regression line.

![image](https://github.com/user-attachments/assets/74c9e032-7719-49e5-b3a5-401c06722b19)

### Model Parameters

- **Intercept (c)**: 2.01816
- **Slope (m)**: 9.91065648

Using the equation of the line \( \text{y} = m \times \text{x} + c \), the model can predict scores for any given number of study hours.

---

## 🌟 Why This Project?

This project showcases the simplicity and power of supervised machine learning. By focusing on a single feature (**Hours**) and a target variable (**Scores**), I demonstrated how to:

- Prepare and preprocess data for modeling.
- Train a Linear Regression model.
- Visualize the relationship between features and targets.
- Make predictions using the trained model.

While this project does not include advanced evaluation metrics like MAE, MSE, or R-squared, it provides a clear and beginner-friendly introduction to supervised machine learning.

---

Thank you for visiting this repository! Feel free to explore, contribute, or share your feedback. 😊
