import abc
import torch
import argparse

from losses import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()

def main():
    xs = []
    ys = []
    loss = loss_10(1, 0.1)
    players_w = [torch.randn(1) for _ in range(2)]
    scale = 1
    norm =  torch.sqrt(players_w[0] ** 2 + players_w[1] ** 2) * scale
    players_w[0] = players_w[0] / norm
    players_w[1] = players_w[1] / norm
    for w in players_w:
        w.requires_grad = True
        w.retain_grad()
    for i in range(10):
        xs.append(float(players_w[0]))
        ys.append(float(players_w[1]))
        grads = torch.stack(grad(loss, players_w))
        H = torch.sum(grads ** 2) / 2
        dh = []
        for g, w in zip(grads, players_w):
            dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
        dh = torch.stack(dh)
        S, A = decompose(grads, players_w)
        Ate = torch.transpose(A, 0, 1) @ grads 
        lamb = torch.sign((grads.view(1,-1) @ dh.view(-1,1)) * (Ate.view(1,-1) @ dh.view(-1, 1)) + 0.1)
        grads = grads + lamb * Ate 
        players_w[0].data.sub_(grads[0] * args.lr)
        players_w[1].data.sub_(grads[1] * args.lr)
        
        # print(players_w)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    # plt.scatter(xs, ys)
    circle1 = plt.Circle((0, 0), 1 / scale, color='r', fill=False)
    plt.gca().add_patch(circle1)
    for i in range(len(xs) - 1):
        plt.arrow(xs[i], ys[i], xs[i + 1] - xs[i], ys[i + 1]-ys[i], shape='full', color='b', lw=1, head_length=0.001, head_width=0.001)
    plt.savefig(f"sga_{args.lr}_{args.lamb}.png", bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    main()