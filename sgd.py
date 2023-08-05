import abc
import torch
import argparse

from losses import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)

args = parser.parse_args()

def main():
    xs = []
    ys = []
    loss = loss_10(1, 0.1)
    players_w = [torch.randn(1) for _ in range(2)]
    norm = torch.sqrt(players_w[0] ** 2 + players_w[1] ** 2)
    players_w[0] = players_w[0] / norm
    players_w[1] = players_w[1] / norm
    for w in players_w:
        w.requires_grad = True
        w.retain_grad()
    for i in range(100):
        grads = grad(loss, players_w)
        S, A = decompose(grads, players_w)
        players_w[0].data.sub_(grads[0] * 0.032)
        players_w[1].data.sub_(grads[1] * 0.032)
        xs.append(float(players_w[0]))
        ys.append(float(players_w[1]))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    # plt.scatter(xs, ys)
    # plt.Circle((0, 0), 1, color='g')
    for i in range(len(xs) - 1):
        plt.arrow(xs[i], ys[i], xs[i + 1] - xs[i], ys[i + 1]-ys[i], shape='full', color='b', lw=1, head_length=0.3, head_width=0.3)
    plt.savefig("sgd.png", bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    main()