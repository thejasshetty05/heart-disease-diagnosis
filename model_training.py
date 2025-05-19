import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the cleaned dataset
df = pd.read_csv(r"data/cleaned_heart.csv")
df = df.drop(columns=["thal"])
# Encode categorical variables
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Standardize numerical columns
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split data into features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "models/trained_model.pkl")
print("Model training complete. Model saved as 'trained_model.pkl'")
