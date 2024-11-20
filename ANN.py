import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the data
def load_data(file_path):
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Shape of data: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Preprocess the data
def preprocess_data(data):
    try:
        # Create a copy of the data
        df = data.copy()
        
        # Convert categorical variables to numeric
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        df['Occupation'] = le.fit_transform(df['Occupation'])
        df['BMI_Category'] = le.fit_transform(df['BMI_Category'])
        
        # Convert Blood Pressure to numeric
        df['Systolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[0]))
        df['Diastolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[1]))
        df = df.drop('Blood_Pressure', axis=1)
        
        # Convert Yes/No to 1/0
        df['Snoring'] = (df['Snoring'] == 'Yes').astype(int)
        
        # Encode the target variable
        df['Sleep_Disorder'] = le.fit_transform(df['Sleep_Disorder'])
        
        print("Data preprocessing completed!")
        return df, le
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None

# Create the neural network model
def create_model(input_dim, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Specify your CSV file path
    file_path = 'Sleep_health_and_lifestyle_dataset.csv'  # Replace with your actual file path
    
    # Load the data
    data = load_data(file_path)
    if data is None:
        return
    
    # Preprocess the data
    df, label_encoder = preprocess_data(data)
    if df is None:
        return
    
    # Define features
    features = ['Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Quality_of_Sleep',
               'Physical_Activity_Level', 'Stress_Level', 'BMI_Category', 'Systolic',
               'Diastolic', 'Heart_Rate', 'Daily_Steps', 'SPO2_Rate', 'Snoring']
    
    # Prepare features and target
    X = df[features].values
    y = df['Sleep_Disorder'].values
    y = to_categorical(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the model
    model = create_model(len(features), y.shape[1])
    
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the model (optional)
    model.save('sleep_disorder_model.h5')
    print("\nModel saved as 'sleep_disorder_model.h5'")

if __name__ == "__main__":
    main()

