import sys, os, json
import numpy as np

def load_model(path="model/model.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("model/model.json not found. Run training first.")
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

def load_data():
	x_data, y_data = load_file("data/data.csv")
	t0, t1, mu, sigma = load_model()
	y_reg_func = lambda x: t0 + t1 * ((x - mu) / sigma)
	return np.array([y_reg_func(x) for x in x_data]),y_data

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



if __name__ == "__main__":

	y,y_hat=load_data()
	mse=MSE(y,y_hat)
	rmse=np.sqrt(mse)
	mae=MAE(y,y_hat)
	print(f"Mean Squared Error (MSE): {mse:.2f}")
	print(f"Root Mean Squared Error (MSE): {rmse:.2f}")
	print(f"Mean Absolute Error (MAE): {mae:.2f}")	