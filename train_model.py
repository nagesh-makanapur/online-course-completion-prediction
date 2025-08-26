import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Hardcoded dataset path
    data_path = "data/online_course_completion.csv"

    # Load dataset
    df = pd.read_csv(data_path)

    # Show columns for debugging
    print("\n✅ Columns in dataset:", list(df.columns))
    print("\n📌 First 5 rows:\n", df.head())

    # Automatically pick the last column as the target
    target_column = df.columns[-1]
    print(f"\n🎯 Using '{target_column}' as the target column.")

    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"\n✅ Model trained successfully. Accuracy: {accuracy:.2f}")

    # Save model
    joblib.dump(model, "course_completion_model.pkl")
    print("\n💾 Model saved as course_completion_model.pkl")

if __name__ == "__main__":
    main()
