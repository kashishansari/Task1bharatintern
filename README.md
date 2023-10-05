# Task1bharatintern
from sklearn.tree import DecisionTreeClassifier

# Define sample data (0 for didn't survive, 1 for survived)
data = [
    [1, 1, 0, 22, 1],  # [class, gender, socio-economic status, age, family size]
    [1, 0, 1, 38, 1],
    [1, 0, 0, 26, 0],
    [1, 0, 1, 35, 1],
    [0, 1, 0, 35, 0],
    [0, 1, 1, 54, 0],
    [0, 1, 1, 2, 3],
    [0, 0, 0, 27, 1]
]

# Split data into features and labels
X = [row[1:] for row in data]
y = [row[0] for row in data]

# Create and train a decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Define user input
user_input = [1, 0, 0, 30,]  # [gender, socio-economic status, age, family size]

# Predict survival outcome
prediction = model.predict([user_input])

if prediction[0] == 1:
    print("According to the model, the person is likely to survive.")
else:
    print("According to the model, the person is less likely to survive.")
