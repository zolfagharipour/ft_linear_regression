import sys, os, json
import numpy as np

def load_model(path="model/model.json"):
    if not os.path.exists(path):
        print("model/model.json not found. Run training first.")
        exit(0)
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # accept either mu/sigma or mu_x/sigma_x if you ever changed names
    mu = float(m.get("mu", m.get("mu_x", 0.0)))
    sigma = float(m.get("sigma", m.get("sigma_x", 1.0)))
    return float(m["theta0"]), float(m["theta1"]), mu, sigma


def load_file(csv_path):
    xs, ys = [], []
    try:
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
    except Exception:
        print(f"Error: cannot read {csv_path}")
        exit(0)
    return np.array(xs, dtype=float), np.array(ys, dtype=float) 

def load_data(csv_path="data/data.csv"):
    x_data, y_data = load_file(csv_path)
    t0, t1, mu, sigma = load_model()
    y_reg_func = lambda x: t0 + t1 * ((x - mu) / sigma)
    return np.array([y_reg_func(x) for x in x_data]),y_data



def R2_score(y, mse):
    var_y = np.var(y)
    return 1 - (mse / var_y)


def MAPE(y, y_hat):
    """Return Mean Absolute Percentage Error."""
    # avoid division by zero by filtering out y == 0
    mask = y != 0
    if not np.any(mask):
        return float("inf")
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask])) * 100


def MSE(y,y_hat):
    mse=0	
    for y1,y2 in zip(y,y_hat):
        mse+=(y1-y2)**2
    return mse/y.size

def MAE(y,y_hat):
    mae=0	
    for y1,y2 in zip(y,y_hat):
        mae+=abs(y1-y2)
    return mae/y.size

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
    y,y_hat=load_data(csv_path)
    mse=MSE(y,y_hat)
    rmse=np.sqrt(mse)
    mae=MAE(y,y_hat)
    r2 = R2_score(y, mse) * 100
    print(f"\n\n\033[1;35mModel Accuracy (R²): \033[1;33m{r2:.2f}%\033[0m")

    print(f"\n\033[1;31mMean Squared Error (MSE): \033[1;33m{mse:.2f}\033[0m")
    print(f"\033[1;32mRoot Mean Squared Error (RMSE): \033[1;33m{rmse:.2f}\033[0m")
    print(f"\033[1;34mMean Absolute Error (MAE): \033[1;33m{mae:.2f}\033[0m")


if __name__ == "__main__":
    main()