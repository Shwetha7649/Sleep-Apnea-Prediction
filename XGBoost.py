import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

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

# Split features and target
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
    
    return X_train, X_test, y_train, y_test

# Train XGBoost model
def train_xgboost(X_train, y_train):
    # Define model parameters
    params = {
        'objective': 'multi:softmax',  # for multi-class classification
        'num_class': len(np.unique(y_train)),  # number of classes
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }
    
    # Create DMatrix with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    
    # Train the model
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)
    
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    predictions = model.predict(dtest)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    return predictions

# Main execution
def main():
    # Read the data
    df = pd.read_csv('C:/Users/91984/Desktop/Sleep_health_and_lifestyle_dataset.csv')  # Replace with your file path
    
    # Preprocess the data
    df_processed = preprocess_data(df)
    
    # Prepare train and test sets
    X_train, X_test, y_train, y_test = prepare_data(df_processed)
    
    # Train the model
    model = train_xgboost(X_train, y_train)
    
    # Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    feature_importance = model.get_score(importance_type='weight')
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")

if __name__ == "__main__":
    main()