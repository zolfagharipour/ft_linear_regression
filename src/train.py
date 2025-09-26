
import json, os
import numpy as np

# J(θ) = (1/(2m)) * sum( (θ0 + θ1 x_i - y_i)^2 )
# ∂J/∂θ0 = (1/m) * sum( (θ0 + θ1 x_i - y_i) )
# ∂J/∂θ1 = (1/m) * sum( (θ0 + θ1 x_i - y_i) * x_i )

def load_xy(csv_path):
    xs, ys = [], []
    with open(csv_path, "r") as f:
        _ = f.readline()
        for line in f:
            try:
                line = line.strip()
                if not line:
                    continue
                x_str, y_str = line.split(",")
                xs.append(float(x_str))
                ys.append(float(y_str))
            except Exception:
                continue
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def standardize(x):
    mu = float(np.mean(x))
    var = float(np.mean((x - mu) ** 2))
    sigma = var ** 0.5
    if sigma < 1e-12:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    return x_scaled, mu, sigma


def train(x, y, alpha=0.05, epochs=2000):
    x, mu, sigma = standardize(x)
    m = x.size
    theta0, theta1 = 0.0, 0.0
    for i in range(epochs):
        yhat = theta0 + theta1 * x
        err  = yhat - y
        d0 = (1.0/m) * np.sum(err)
        d1 = (1.0/m) * np.sum(err * x)
        theta0 -= alpha * d0
        theta1 -= alpha * d1
    return theta0, theta1, mu, sigma

def main():
    x, y = load_xy("data/data.csv")
    t0, t1, mu, sigma = train(x, y)
    os.makedirs("model", exist_ok=True)
    with open("model/model.json", "w") as f:
        json.dump({"theta0": t0, "theta1": t1, "mu": mu, "sigma": sigma}, f)
    print(f"Learned: theta0={t0:.6f}, theta1={t1:.6f} -> saved to model/model.json")

if __name__ == "__main__":
    main()
