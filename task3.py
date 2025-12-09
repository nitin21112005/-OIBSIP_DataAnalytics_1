import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def main():
    # 1. Load data
    df = pd.read_csv("car_data.csv")
    print("Dataset shape:", df.shape)
    print(df.head())

    # 2. Basic info
    print("\nInfo:")
    print(df.info())

    # 3. Simple EDA plot (price distribution)
    plt.figure()
    sns.histplot(df["Selling_Price"], kde=True)
    plt.xlabel("Selling Price")
    plt.title("Distribution of Car Selling Price")
    plt.tight_layout()
    plt.savefig("price_distribution.png", dpi=300)
    plt.close()

    # 4. Feature/target split
    X = df.drop(["Selling_Price", "Car_Name"], axis=1)
    y = df["Selling_Price"]

    # 5. Identify numeric & categorical features
    numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Build model
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    clf = Pipeline(steps=[("preprocessor", preprocessor),
                         ("model", model)])

    # 8. Train
    clf.fit(X_train, y_train)

    # 9. Predict
    y_pred = clf.predict(X_test)

    # 10. Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 Score: {r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

    # 11. Plot Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.close()

    print("Model training complete. Plots saved as PNG files in the project folder.")

if __name__ == "__main__":
    main()
