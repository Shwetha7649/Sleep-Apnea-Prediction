import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier  # Make sure to install XGBoost if you haven't

# Load dataset
data = pd.read_csv("C:/Users/91984/Desktop/Sleep_health_and_lifestyle_dataset.csv")

# Check for missing values and fill only numeric columns with mean, keeping categorical columns intact
if data.isnull().sum().any():
    print("Missing values detected. Filling missing values...")
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode categorical variables if present
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Check the column names to identify the target column
print("Column names:", data.columns)

# Define the target column
target_column = 'Sleep_Disorder'  # Adjust if your dataset has a different target name

# Define the features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'ANN': MLPClassifier(max_iter=300),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(),
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print('-' * 50)
