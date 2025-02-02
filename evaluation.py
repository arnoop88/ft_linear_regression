import csv
import numpy as np

def load_data(filename):
    kms = []
    prices = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            kms.append(float(row['km']))
            prices.append(float(row['price']))
    return np.array(kms), np.array(prices)

def load_model(filename="data/model.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
        theta0 = float(lines[0].strip())
        theta1 = float(lines[1].strip())
        km_mean = float(lines[2].strip())
        km_std = float(lines[3].strip())
    return theta0, theta1, km_mean, km_std

def predict_price(km, theta0, theta1, km_mean, km_std):
    km_normalized = (km - km_mean) / km_std
    return theta0 + theta1 * km_normalized

def compute_metrics(actual, predicted):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Squared Error (MSE)
    mse = np.mean((actual - predicted) ** 2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mae, rmse, r2

def main():
    # Load data and model
    try:
        kms, prices = load_data("data/data.csv")
        theta0, theta1, km_mean, km_std = load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Generate predictions
    predictions = np.array([predict_price(km, theta0, theta1, km_mean, km_std) for km in kms])

    # Calculate metrics
    mae, rmse, r2 = compute_metrics(prices, predictions)

    # Print results
    print("="*40)
    print(f"{'Model Evaluation Metrics':^40}")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print("="*40)
    print("Interpretation:")
    print("- MAE: Average error in price units")
    print("- RMSE: Weighted average error (punishes large errors)")
    print("- R²: 1 = perfect fit, 0 = baseline model")

if __name__ == "__main__":
    main()