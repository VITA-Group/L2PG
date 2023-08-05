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
    
    loss = loss_quadratic(1, S1=0.6805, S2=-0.3161, M12=1.6537, M21=0.1245, b1=0, b2=0)
    # loss = loss_quadratic(1, S1=0.5, S2=0.5, M12=1, M21=-1, b1=0, b2=0)
   
    def r(n):
        x = torch.cos(torch.FloatTensor([n])).cuda()
        y = torch.sin(torch.FloatTensor([n])).cuda()
        return [x, y]
    players_w = r(np.random.rand() * 1000)
    initial_w = list(players_w)

    # print(players_w)
    # assert False
    # players_w = [torch.tensor(2).float() for _ in range(2)]
    
    scale = 1
    # norm =  torch.sqrt(players_w[0] ** 2 + players_w[1] ** 2) * scale
    # players_w[0] = players_w[0] # / norm
    # players_w[1] = players_w[1] # / norm
    initial_w = list(players_w)
    lrs = []
    counts = []
    for lr in np.arange(0.02, 2, 0.02):
        acount = 0
        initial_w = [r(2 ** int(_)) for _ in [3]]
        print(initial_w)
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
                grads = torch.stack(grad(loss, players_w)) 
                H = torch.sum(grads ** 2) / 2
                dh = []
                for g, w in zip(grads, players_w):
                    dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                dh = torch.stack(dh)
                S, A = decompose(grads , players_w)
                Ate = torch.transpose(A, 0, 1) @ grads 
                lamb = args.lamb
                grads = (grads + lamb * Ate).detach() 
                losses.extend(loss(players_w))
                players_w[0] = players_w[0] - grads[0] * lr  
                players_w[1] = players_w[1] - grads[1] * lr 

                if torch.mean(torch.abs(torch.stack(losses[-20:]))) < 0.01:
                    break
                else:
                    count += 1
                # print(players_w)
            # print(lr, count)
            acount += count
        lrs.append(lr)
        counts.append(acount / len(initial_w))
    import matplotlib.pyplot as plt
    with open("sga_quadratic.list", "w") as f:
        f.write(','.join(list(map(str, lrs))) + "\n")
        f.write(','.join(list(map(str, counts))) + "\n")
    plt.plot(lrs, counts)
    plt.savefig("sga_line_align.png")
    plt.close()
if __name__ == "__main__":
    main()