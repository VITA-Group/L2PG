import abc
import torch
import argparse
import numpy as np
from losses import *
import torch.nn as nn
from networks import RNNOptimizer
import tree

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument('--random-lr', action="store_true")
args = parser.parse_args()

def detach(x):
    x.detach_()
    x.requires_grad = True
    return x


def init_w():
    def r(n):
        x = torch.cos(torch.FloatTensor([n])).cuda()
        y = torch.sin(torch.FloatTensor([n])).cuda()
        return [x, y]
    players_w = r(np.random.rand() * 1000)
    initial_w = list(players_w)
    return initial_w

n_player = 2

def main():
    cl = [5, 10, 20]
    count = 0
    loss = loss_quadratic(1, 0.1)
    optimizer = RNNOptimizer(True, 20, 10, False).cuda()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), 0.001)

    for epoch in range(150):
        S1 = torch.randn((args.n_action, args.n_action)).cuda()
        S2 = torch.randn((args.n_action, args.n_action)).cuda()
        M12 = torch.randn((args.n_action, args.n_action)).cuda()
        M21 = torch.randn((args.n_action, args.n_action)).cuda()
        b1 = torch.zeros(args.n_action).cuda()
        b2 = torch.zeros(args.n_action).cuda()
        loss = loss_quadratic(1, S1, S2, M12, M21, b1, b2)
        epoch_w = init_w()
        for w in epoch_w:
            w.requires_grad = True
            w.retain_grad()
        hiddens = [[torch.zeros(w.numel() * n_player, 20).cuda()]]
        cells = [[torch.zeros(w.numel() * n_player, 20).cuda()]]
        meta_loss = 0
        # print(epoch_w)
        for iterations in range(50):
            
            grads = torch.cat(grad(loss, epoch_w), 0) # (np * na) x 1
            S, A = decompose(torch.stack(grad(loss, epoch_w)), epoch_w) # (np * na) x (np * na) 
            Ates = torch.transpose(A, 0, 1) @ grads
            Ss = S @ grads
            obs = torch.cat([grads.view(-1, 1), Ates.view(-1, 1), Ss.view(-1, 1)], dim=1)
            new_hs = []
            new_cs = []
            
            update, new_h, new_c = optimizer(obs.detach(), hiddens[0], cells[0])
            if iterations < 10:
                # print(update)
                # print(epoch_w)
                pass
            # print(iterations, epoch_w, update)
            epoch_w[0] = epoch_w[0] - update[0,0] * grads[0] - update[0,1] * Ates[0] - update[0,2] * Ss[0]
            epoch_w[1] = epoch_w[1] - update[1,0] * grads[1] - update[1,1] * Ates[1] - update[1,2] * Ss[1]
            new_hs.append(new_h)
            new_cs.append(new_c)
            hiddens = new_hs
            cells = new_cs

            if (iterations + 1) % cl[epoch // 50] == 0:
                meta_loss.backward()
                for p in optimizer.parameters():
                    # print(p.grad)
                    pass
                torch.nn.utils.clip_grad_norm_(optimizer.parameters(), 1)
                meta_optimizer.step()
                optimizer.zero_grad()
                print(meta_loss.item())
                meta_loss = 0
                hiddens = tree.map_structure(detach, hiddens)
                cells = tree.map_structure(detach, cells)
                epoch_w = tree.map_structure(detach, epoch_w)
                del Ates
                del grads
            else:
                meta_loss += 1 / 2 * torch.sum(grads ** 2)
            
            if meta_loss > 1e8:
                break

        torch.save(optimizer.state_dict(), 'optimizer_quadratic.pkl')

if __name__ == "__main__":
    main()