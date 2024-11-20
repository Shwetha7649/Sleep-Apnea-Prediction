import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data preprocessing
def preprocess_data(df):
    # Create a copy of the dataframe
    df = df.copy()
    
    # Handle Blood Pressure column
    def split_blood_pressure(bp_string):
        systolic, diastolic = map(int, bp_string.split('/'))
        return systolic, diastolic

    # Split Blood_Pressure into systolic and diastolic
    df[['Systolic_BP', 'Diastolic_BP']] = pd.DataFrame(
        df['Blood_Pressure'].apply(split_blood_pressure).tolist(),
        index=df.index
    )
    
    # Drop original Blood_Pressure column
    df = df.drop('Blood_Pressure', axis=1)
    
    # Convert categorical variables to numerical using Label Encoding
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Occupation', 'BMI_Category', 'Sleep_Disorder']
    
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    
    # Convert Yes/No to 1/0 for binary columns
    binary_columns = ['Snoring']
    for column in binary_columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})
    
    # Ensure all columns are numeric
    numeric_columns = ['Age', 'Sleep_Duration', 'Quality_of_Sleep', 
                      'Physical_Activity_Level', 'Stress_Level', 'Heart_Rate',
                      'Daily_Steps', 'SPO2_Rate']
    
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Fill any missing values (if any)
    df = df.fillna(df.mean())
    
    return df

# Split features and target and scale the features
def prepare_data(df):
    # Define feature columns (all except Sleep_Disorder)
    feature_columns = [col for col in df.columns if col != 'Sleep_Disorder']
    
    # Separate features and target
    X = df[feature_columns]
    y = df['Sleep_Disorder']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

# Train SVM model
def train_svm(X_train, y_train):
    # Create and train SVM model
    svm_model = SVC(
        kernel='rbf',  # Radial Basis Function kernel
        C=1.0,        # Regularization parameter
        gamma='scale', # Kernel coefficient
        random_state=42
    )
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    return svm_model

# Evaluate model
def evaluate_model(model, X_test, y_test, feature_columns):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return predictions

# Main execution
def main():
    # Read the data
    df = pd.read_csv("C:/Users/91984/Desktop/Sleep_health_and_lifestyle_dataset.csv")  # Replace with your file path
    
    # Preprocess the data
    df_processed = preprocess_data(df)
    
    # Prepare train and test sets
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns = prepare_data(df_processed)
    
    # Train the model
    print("Training SVM model...")
    model = train_svm(X_train_scaled, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    predictions = evaluate_model(model, X_test_scaled, y_test, feature_columns)
    
    # Print model parameters
    print("\nModel Parameters:")
    print(f"Kernel: {model.kernel}")
    print(f"C: {model.C}")
    print(f"Gamma: {model.gamma}")
    
    # Calculate and print accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Print support vectors info
    print(f"\nNumber of Support Vectors: {model.n_support_}")
    print(f"Number of Support Vectors per Class: {model.support_}")

if __name__ == "__main__":
    main()