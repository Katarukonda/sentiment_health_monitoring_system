# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Generate Synthetic Data
np.random.seed(0)
n_samples = 500

data = pd.DataFrame({
    'age': np.random.randint(20, 60, n_samples),
    'heart_rate': np.random.randint(60, 180, n_samples),
    'calories': np.random.randint(1500, 3000, n_samples),
    'risk': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 70% low risk, 30% high risk
})

# Display first few rows of data
print("Synthetic Data Sample:")
print(data.head())

# Step 2: Visualize the Data
plt.figure(figsize=(10, 5))

# Age vs Risk
plt.subplot(1, 2, 1)
sns.histplot(data=data, x="age", hue="risk", multiple="stack", kde=True)
plt.title("Age vs Health Risk")

# Heart Rate vs Risk
plt.subplot(1, 2, 2)
sns.histplot(data=data, x="heart_rate", hue="risk", multiple="stack", kde=True)
plt.title("Heart Rate vs Health Risk")

plt.tight_layout()
plt.show()

# Step 3: Prepare Data for Machine Learning
# Splitting the dataset into features and target
X = data[['age', 'heart_rate', 'calories']]
y = data['risk']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Train a Machine Learning Model
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)

# Step 5: Visualize the Prediction Results
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.show()

# Step 6: Define a Function for Health Risk Prediction
def predict_health_risk(age, heart_rate, calories):
    # Prepare data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'heart_rate': [heart_rate],
        'calories': [calories]
    })
    risk = model.predict(input_data)[0]
    return "High Risk" if risk == 1 else "Low Risk"

# Step 7: Get User Input and Predict Health Risk
try:
    age = int(input("Enter your age: "))
    heart_rate = int(input("Enter your heart rate: "))
    calories = int(input("Enter the number of calories consumed: "))
    
    prediction = predict_health_risk(age, heart_rate, calories)
    print(f"Health Risk Prediction for age {age}, heart rate {heart_rate}, and calories {calories}: {prediction}")

except ValueError:
    print("Please enter valid integer values for age, heart rate, and calories.")
