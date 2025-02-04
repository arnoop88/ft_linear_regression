import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.001
NUM_ITERATIONS = 10000

def load_data(filename):
    kms = []
    prices = []
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                kms.append(float(row['km']))
                prices.append(float(row['price']))
        return np.array(kms), np.array(prices)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        exit()
    except PermissionError:
        print(f"Error: Permission denied to open {filename}")
        exit()

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std

def compute_cost(theta0, theta1, kms, prices):
    m = len(prices)
    predictions = theta0 + theta1 * kms
    errors = predictions - prices
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost

def gradient_descent(kms, prices, theta0, theta1, learning_rate, iterations):
    m = len(prices)
    cost_history = []

    for i in range(iterations):
        predictions = theta0 + theta1 * kms
        errors = predictions - prices

        grad_theta0 = (1 / m) * np.sum(errors)
        grad_theta1 = (1 / m) * np.sum(errors * kms)

        theta0 = theta0 - learning_rate * grad_theta0
        theta1 = theta1 - learning_rate * grad_theta1

        cost = compute_cost(theta0, theta1, kms, prices)
        cost_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}")

    return theta0, theta1, cost_history

def save_model(theta0, theta1, km_mean, km_std, filename):
    try:
        with open(filename, 'w') as f:
            f.write(f"{theta0}\n{theta1}\n{km_mean}\n{km_std}\n")
        print(f"Model and normalization parameters saved to {filename}")
    except PermissionError:
        print(f"Error: Permission denied to save {filename}")
        return

def plot_results(kms, prices, theta0, theta1, km_mean, km_std, cost_history):
    os.makedirs('graphs', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(kms, prices, color='blue', alpha=0.5, label='Actual Prices', marker='o')
    
    # Regression line
    x_min, x_max = np.min(kms), np.max(kms)
    x_line = np.linspace(x_min, x_max, 100)
    x_line_normalized = (x_line - km_mean) / km_std
    y_line = theta0 + theta1 * x_line_normalized
    
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
    
    # Add labels and title
    plt.xlabel('Mileage (km)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title('Car Price vs Mileage with Linear Regression', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save and close
    plt.tight_layout()
    plt.savefig('graphs/price_vs_km_plot.png', dpi=300)
    plt.close()
    print("Saved visualization to graphs/price_vs_km_plot.png")
    plt.plot(range(NUM_ITERATIONS), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.savefig("graphs/cost_plot.png")
    plt.close()
    print("Saved visualization to graphs/cost_plot.png")

def main():
    # Load dataset
    kms, prices = load_data("data/data.csv")
    
    # Normalize km values
    kms_normalized, km_mean, km_std = normalize_data(kms)

    # Initialize model parameters
    theta0 = 0.0
    theta1 = 0.0

    # Train using gradient descent
    theta0, theta1, cost_history = gradient_descent(kms_normalized, prices, theta0, theta1, LEARNING_RATE, NUM_ITERATIONS)

    # Save the model parameters
    save_model(theta0, theta1, km_mean, km_std, "data/model.txt")

    plot_results(kms, prices, theta0, theta1, km_mean, km_std, cost_history)

if __name__ == "__main__":
    main()
