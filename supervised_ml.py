from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

# 1. Prepare the dataset using Python lists
hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

# 2. Separate  input features (X) and target value (y)
X = [[hour] for hour in hours]
y = scores

# 3. Create the Linear Regression model
model = LinearRegression()

# 4. Train the model
model.fit(X, y)

# 5. Get input from the user
study_hours = float(input("Enter hours studied: "))

# 6. Predict the examination score 
prediction = model.predict([[study_hours]])

# 7. Display the prediction to two decimal places
print(f"Predicted exam score: {prediction[0]: .2f}")

# Bonus task: Plot the relationship between hours and scores
plt.scatter(hours, scores, label="Actual scores")
plt.plot(hours, model.predict(X), label="Regression line")
plt.xlabel("Hours Studied") 
plt.ylabel("Examination Score")
plt.title("Study Hours vs Examination Scores")
plt.legend()
plt.grid(True)
plt.show()
