import abc
import torch
import argparse
import numpy as np
import tree
from losses import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)



args = parser.parse_args()
def r(n):
    x = torch.cos(torch.FloatTensor([n])).cuda()
    y = torch.sin(torch.FloatTensor([n])).cuda()
    return [x, y]

def main():
    loss, coef = loss_quadratic(1, 0.9833735514226325,0.6604625073318571,-1.0811084089987542,0.0011816860028365868,-0.04818994422299865,-0.5598408772961735)

    for lr in [0.5]:
        for lamb in [0.4]:
            ws = []
            count = 0
            initial_w = [r(2**int(_)) for _ in [1]]
            print(initial_w)

            players_w = initial_w[0]
            print(players_w)
            losses = []
            ws.append(list(players_w))
            xs = []
            ys = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
            count = 0
            grads_ = []
            while count < 100:
                grads = grad(loss, players_w)
                grads_.append(grads.detach())
                # print(grads)
                H = torch.sum(grads ** 2) / 2
                dh = []
                for g, w in zip(grads, players_w):
                    dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                dh = torch.stack(dh)
                S, A = decompose(grads , players_w)
                Ate = torch.transpose(A, 0, 1) @ grads 
                grads = (grads + lamb * Ate * torch.sign((grads.view(1,-1) @ dh.view(-1,1)) * (Ate.view(1,-1) @ dh.view(-1, 1)) + 0.1)).view(-1).detach() 
                # print(grads)
                losses.extend(loss(players_w))
                # print(losses)
                ws.append(list(players_w))
                players_w[0] = players_w[0] - grads[0] * lr / 2
                players_w[1] = players_w[1] - grads[1] * lr / 2
                
                if torch.mean(torch.norm(torch.stack(grads_[-10:]), dim=1)) < 0.001:
                    break
                else:
                    count += 1

            ws = tree.map_structure(lambda x: float(x), ws)
            ws = np.array(ws)
            import pandas as pd
            pd.DataFrame({'p1': ws[:, 0], 'p2': ws[:, 1]}).to_csv(f'sga_path.csv')
            losses = tree.map_structure(lambda x: x.detach().cpu().numpy(), losses)
            
            plt.figure(figsize=(4, 4))

            for i in range(len(ws) - 1):
                plt.arrow(ws[i, 0], ws[i, 1], ws[i + 1, 0] - ws[i, 0], ws[i+1, 1]-ws[i, 1], shape='full', color='b', lw=1, head_length=0.01, head_width=0.01)
            plt.title(f'count={count}')
            plt.savefig(f"sga/sga_path_lr{lr}_lambda{lamb}.png", bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(4, 4))
            
            plt.plot(list(range(len(losses) // 2)), losses[::2])
            plt.plot(list(range(len(losses) // 2)), losses[1::2])
            plt.title(f'count={count}')
            plt.savefig(f"sga/sga_loss_lr{lr}_lambda{lamb}.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()