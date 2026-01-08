
import numpy as np
import matplotlib.pyplot as plt
import os

from optimizers import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

a, b = 1.0, 5.0

def loss_fn(params):
    x, y = params
    return a * x**2 + b * y**2

def grad_fn(params):
    x, y = params
    return np.array([2 * a * x, 2 * b * y])

init_params = np.array([5.0, 5.0])
num_iters = 100

optimizers = {
    "SGD": SGD(lr=0.05),
    "Momentum": Momentum(lr=0.05),
    "Nesterov": Nesterov(lr=0.05),
    "AdaGrad": AdaGrad(lr=0.5),
    "RMSProp": RMSProp(lr=0.1),
    "Adam": Adam(lr=0.1),
}

histories = {}

for name, opt in optimizers.items():
    params = init_params.copy()
    opt.reset()
    traj = [params.copy()]
    losses = []

    for _ in range(num_iters):
        if name == "Nesterov":
            params = opt.step(params, grad_fn)
        else:
            params = opt.step(params, grad_fn(params))

        traj.append(params.copy())
        losses.append(loss_fn(params))

    histories[name] = {"trajectory": np.array(traj), "losses": np.array(losses)}

plt.figure(figsize=(8, 5))
for name, h in histories.items():
    plt.plot(h["losses"], label=name)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_vs_iteration.png"), dpi=300)
plt.close()

x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = a * X**2 + b * Y**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30)
for name, h in histories.items():
    traj = h["trajectory"]
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3, label=name)
plt.scatter(0, 0, c="red", s=100, label="Minimum")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "optimizer_trajectories.png"), dpi=300)
plt.close()
