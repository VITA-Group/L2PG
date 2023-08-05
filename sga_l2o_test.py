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
parser.add_argument('--checkpoint', type=str)

args = parser.parse_args()

def detach(x):
    x.detach_()
    x.requires_grad = True
    return x

n_player = 2

def main():
    # loss = loss_seesaw(1, 0.1)
    def r(n):
        x = torch.cos(torch.FloatTensor([n])).cuda()
        y = torch.sin(torch.FloatTensor([n])).cuda()
        return [x, y]
    optimizer = RNNOptimizer(True, 20, 10, False).cuda()
    # meta_optimizer = torch.optim.Adam(optimizer.parameters(), 0.01)
    optimizer.load_state_dict(torch.load(args.checkpoint))
    lrs = []
    counts = []
    for lr in [1] * 10:
        S1 = torch.randn((args.n_action, args.n_action)).cuda()
        S2 = torch.randn((args.n_action, args.n_action)).cuda()
        M12 = torch.randn((args.n_action, args.n_action)).cuda()
        M21 = torch.randn((args.n_action, args.n_action)).cuda()
        print(S1, S2, M12, M21)
        b1 = torch.zeros(args.n_action).cuda()
        b2 = torch.zeros(args.n_action).cuda()
        loss = loss_quadratic(1, S1, S2, M12, M21, b1, b2)
        acount = 0
        initial_w = [r(_ * 1000) for _ in np.arange(0,10,0.8)]
        for wi in initial_w:
            players_w = wi
            losses = []
            xs = []
            ys = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
                w.cuda()

            hiddens = [[torch.zeros(w.numel() * n_player, 20).cuda()]]
            cells = [[torch.zeros(w.numel() * n_player, 20).cuda()]]
            count = 0

            while count < 250:
                grads = torch.stack(grad(loss, players_w)).cuda()
        
                S, A = decompose(grads, players_w) # (np * na) x (np * na) 
                Ate = torch.transpose(A, 0, 1) @ grads
                Ss = S @ grads

                obs = torch.cat([grads.view(-1, 1), Ate.view(-1, 1), Ss.view(-1, 1)], dim=1)
                new_hs = []
                new_cs = []
                
                
                with torch.no_grad():
                    update, new_h, new_c = optimizer(obs.detach(), hiddens[0], cells[0])
                
                # grads = (grads + update[:,1].view(2,1) * Ate).detach() 
                losses.extend(loss(players_w))

                players_w[0] = players_w[0] - grads[0] * update[0,0] - update[0,1] * Ate[0] - update[0,2] * Ss[0]
                players_w[1] = players_w[1] - grads[1] * update[1,0] - update[1,1] * Ate[1] - update[1,2] * Ss[1]
                # print(players_w)
                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs
                if torch.mean(torch.abs(torch.stack(losses[-20:]))) < 0.01:
                    break
                else:
                    count += 1
            
            acount += count
            # assert False
        lrs.append(lr)
        counts.append(acount / len(initial_w))
        # assert False
    
    with open("sga_l2o_1.0_quadratic.list", "w") as f:
        f.write(','.join(list(map(str, lrs))) + "\n")
        f.write(','.join(list(map(str, counts))) + "\n")

    import matplotlib.pyplot as plt
    plt.plot(lrs, counts)
    plt.savefig("sga_line_l2o_quadratic.png")
    plt.close()

if __name__ == "__main__":
    main()