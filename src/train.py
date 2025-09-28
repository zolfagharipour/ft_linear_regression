
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# J(θ) = (1/(2m)) * sum( (θ0 + θ1 x_i - y_i)^2 )
# ∂J/∂θ0 = (1/m) * sum( (θ0 + θ1 x_i - y_i) )
# ∂J/∂θ1 = (1/m) * sum( (θ0 + θ1 x_i - y_i) * x_i )

def load_xy(csv_path):
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


def standardize(x):
    mu = float(np.mean(x))
    var = float(np.mean((x - mu) ** 2))
    sigma = var ** 0.5
    if sigma < 1e-12:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    return x_scaled, mu, sigma

def Plot_Graph(x, y, yhat, theta0, theta1, epoch, xmin, xmax, m, err):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Epoch #{epoch:4d}")
    ax.scatter(x, y, color='blue', label='data')
    ax.plot([xmin, xmax], [yhat[list(x).index(xmin)], yhat[list(x).index(xmax)]], color='red', label='regression', ls='--')
    ax.legend(loc=0)
    plt.savefig(f"plots/graph_{epoch}.png")
    plt.close(fig)
    im = Image.open(f"plots/graph_{epoch}.png")
    return im



def train(x, y, alpha=0.05, epochs=100):
    os.makedirs("plots", exist_ok=True)
    standard_x, mu, sigma = standardize(x)
    m = standard_x.size
    theta0, theta1 = 0.0, 0.0
    xmin,xmax,images=min(x),max(x),[]
    for i in range(epochs):
        yhat = theta0 + theta1 * standard_x
        err  = yhat - y
        d0 = (1.0/m) * np.sum(err)
        d1 = (1.0/m) * np.sum(err * standard_x)
        theta0 -= alpha * d0
        theta1 -= alpha * d1
        im = Plot_Graph(x, y, yhat, theta0, theta1, i, xmin, xmax, m, err)
        images.append(im)

    os.rename(f"plots/graph_{epochs-1}.png","plot.png")
    images[0].save('epochs.gif',save_all=True,append_images=images[1:], optimize=False, duration=500, loop=0)


    return theta0, theta1, mu, sigma

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
    
    csv_file = sys.argv[1]
    x, y = load_xy(csv_file)
    t0, t1, mu, sigma = train(x, y)
    os.makedirs("model", exist_ok=True)
    with open("model/model.json", "w") as f:
        json.dump({"theta0": t0, "theta1": t1, "mu": mu, "sigma": sigma}, f)
    print(f"Learned: theta0={t0:.6f}, theta1={t1:.6f} -> saved to model/model.json")


if __name__ == "__main__":
    main()
