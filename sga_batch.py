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

def load_games_list(args):
    lines = open(args.game_list).readlines()
    array_lines = []
    for line in lines:
        array_lines.append(np.array(list(map(float, line.strip().split(',')))))
    return array_lines

def r(n):
    x = torch.cos(torch.FloatTensor([n])).cuda()
    y = torch.sin(torch.FloatTensor([n])).cuda()
    return [x, y]

def main():
    loss, coef = loss_quadratic(1, S1=0.5, S2=0.5, M12=0.5, M21=0.5, b1=0, b2=0)

    for lr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]:
        for lamb in [0.1, 0.2, 0.5, 0.7, 1, 1.5, 2]:
            ws = []
            count = 0
            initial_w = [r(2**int(_)) for _ in [3]]
            print(initial_w)

            players_w = initial_w[0]
            losses = []
            ws.append(list(players_w))
            xs = []
            ys = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
            count = 0
            while count < 10000:
                grads = torch.stack(grad(loss, players_w)) 
                H = torch.sum(grads ** 2) / 2
                dh = []
                for g, w in zip(grads, players_w):
                    dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                dh = torch.stack(dh)
                S, A = decompose(grads , players_w)
                Ate = torch.transpose(A, 0, 1) @ grads 
                grads = (grads + lamb * Ate * torch.sign((grads.view(1,-1) @ dh.view(-1,1)) * (Ate.view(1,-1) @ dh.view(-1, 1)) + 0.1)).detach() 
                # print(grads)
                losses.extend(loss(players_w))
                # print(losses)
                ws.append(list(players_w))
                players_w[0] = players_w[0] - grads[0] * lr / 2
                players_w[1] = players_w[1] - grads[1] * lr / 2
                
                if torch.mean(torch.abs(torch.stack(losses[-20:]))) < 0.001:
                    break
                else:
                    count += 1

            ws = tree.map_structure(lambda x: float(x), ws)
            ws = np.array(ws)

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