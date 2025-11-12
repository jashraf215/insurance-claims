from src.preprocessing import load_and_clean_data
from src.models import train_logistic_model


def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data("data/insurance_claims.csv")
    print(f"Data shape after cleaning: {df.shape}")

    print("\nTraining Logistic Regression model...")
    model, preprocessor, selector = train_logistic_model(df)
    print("\nModel training complete.")


if __name__ == "__main__":
    main()
