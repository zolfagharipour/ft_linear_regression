import json, os

def load_model(path="model/model.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"theta0": 0.0, "theta1": 0.0, "mu_x": 0.0, "sigma_x": 1.0, "scaler": "standard"}

def main():
    m = load_model()
    t0 = float(m.get("theta0", 0.0))
    t1 = float(m.get("theta1", 0.0))
    mu = float(m.get("mu", 0.0))
    sigma = float(m.get("sigma", 1.0))

    print(f"Loaded θ: theta0={t0:.6f}, theta1={t1:.6f}")
    while True:
        s = input("Enter mileage (or 'q' to quit): ").strip().lower()
        if s in {"q", "quit", "exit"}:
            break
        try:
            x = float(s)
        except ValueError:
            print("Please enter a number.")
            continue

        x_scaled = (x - mu) / sigma
        yhat = t0 + t1 * x_scaled
        print(f"Estimated price: {yhat:.2f}")

if __name__ == "__main__":
    main()
