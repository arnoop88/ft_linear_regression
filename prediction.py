def load_model(filename="data/model.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
        theta0 = float(lines[0].strip())
        theta1 = float(lines[1].strip())
        km_mean = float(lines[2].strip())
        km_std = float(lines[3].strip())
    return theta0, theta1, km_mean, km_std

def predict_price(km, theta0, theta1, km_mean, km_std):
    # Normalize the km value before prediction
    km_normalized = (km - km_mean) / km_std
    return theta0 + theta1 * km_normalized

def main():
    theta0, theta1, km_mean, km_std = load_model()

    try:
        km = float(input("Enter the car's km: "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    price = predict_price(km, theta0, theta1, km_mean, km_std)
    print(f"Predicted price for {km} km is: ${price:.2f}")

if __name__ == "__main__":
    main()
