import abc
import torch
import argparse
import numpy as np
from losses import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()

def main():
    
    loss = loss_sym(1, 0.1)
    def r(n):
        x = torch.FloatTensor([n]).cuda()
        y = 4 / x
        return [x, y]
    players_w = [r(np.random.rand() * 100000) for _ in range(20)]

    initial_w = list(players_w)
    lrs = []
    counts = []
    for lr in np.arange(0.02, 2, 0.02):
        acount = 0
        initial_w = [r(2**int(_)) for _ in np.arange(0,11,1)]
        for wi in initial_w:
            players_w = wi
            losses = []
            xs = []
            ys = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
            count = 0
            prev_grads = [0, 0]
            momentum = [0, 0]
            while count < 250:
                # xs.append(float(players_w[0]))
                # ys.append(float(players_w[1]))
                # print(players_w)
                grads = torch.stack(grad(loss, players_w)) 
                # print(grads.shape)
                if count > 0:
                    current_grads = [2 * grads[0] - prev_grads[0], 2 * grads[1] - prev_grads[1]]
                else:
                    current_grads = [grads[0], grads[1]]
                # current_grads = [grads[0], grads[1]]
                prev_grads = grads
                losses.extend(loss(players_w))
                # print(losses)
                players_w[0].data.sub_(current_grads[0] * lr / 2)
                players_w[1].data.sub_(current_grads[1] * lr / 2)
                if torch.mean(torch.abs(torch.stack(losses[-20:]))) < 0.01:
                    break
                else:
                    count += 1
                # print(players_w)
            # print(lr, count)
            acount += count
            # assert False
        lrs.append(lr)
        counts.append(acount / len(initial_w))
    print(lrs)
    print(counts)
    import matplotlib.pyplot as plt
    with open("omd.list", "w") as f:
        f.write(','.join(list(map(str, lrs))) + "\n")
        f.write(','.join(list(map(str, counts))) + "\n")
    plt.plot(lrs, counts)
    plt.savefig("omd.png")
    plt.close()
if __name__ == "__main__":
    main()