# ml_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving/loading models
import os

def train_and_save_model(data_filename="match_data.csv", model_filename="betting_model.pkl"):
    """
    Loads data, trains a Logistic Regression model, and saves the trained model.

    Args:
        data_filename (str): The name of the CSV file containing the data.
        model_filename (str): The name of the file to save the trained model.
    """
    if not os.path.exists(data_filename):
        print(f"Error: Data file '{data_filename}' not found. Please run data_generator.py first.")
        return

    print(f"Loading data from {data_filename}...")
    df = pd.read_csv(data_filename)
    print("Data loaded successfully.")
    print(df.head())

    # Define features (X) and target (y)
    features = ['team_a_form', 'team_b_form', 'team_a_win_rate', 'team_b_win_rate']
    X = df[features]
    y = df['outcome']

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Initialize and train the Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_filename)
    print(f"Trained model saved to {model_filename}")

if __name__ == "__main__":
    train_and_save_model()