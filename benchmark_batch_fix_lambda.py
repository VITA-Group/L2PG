import abc
import torch
import argparse
import numpy as np
from losses import *
from networks import RNNOptimizer
import matplotlib.pyplot as plt
import tree

parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--game_list", type=str)
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--print_update", action="store_true")
parser.add_argument("--type", type=str)


args = parser.parse_args()

def detach(x):
    x.detach_()
    x.requires_grad = True
    return x
n_player = 2

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

from tqdm import tqdm
def main():
    
    lines, lambdas = load_games_list(args)
    players_w = [r(np.random.rand() * 1000) for _ in range(10)]
    initial_w = list(players_w)
    counts = []
    
    for idx, loss_line in tqdm(enumerate(lines)):
        ws = []
        updates = []
        grads_ = []
        Ags = []
        Sgs = []
        loss, _ = loss_quadratic(1, *list(loss_line))
        for lr in [args.lr]:
            acount = 0
            initial_w = [r(2**int(_)) for _ in range(1, 5)]
            for wi in initial_w:
                players_w = wi
                ws.append(list(wi))
                losses = []
                for w in players_w:
                    w.requires_grad = True
                    w.retain_grad()
                    w.cuda()

                count = 0

                while count < 1000:
                    grads = torch.stack(grad(loss, players_w)).cuda()
                    grads_.append(grads)
                    S, A = decompose(grads, players_w) # (np * na) x (np * na) 
                    Ate = torch.transpose(A, 0, 1) @ grads
                    Ss = S @ grads
                    
                    players_w[0] = players_w[0] - grads[0] * lr - Ate[0] * args.lamb * lr 
                    players_w[1] = players_w[1] - grads[1] * lr - Ate[1] * args.lamb * lr

                    if args.type == 'conopt':
                        players_w[0] = players_w[0] - Ss[0] * args.lamb * lr 
                        players_w[1] = players_w[1] - Ss[1] * args.lamb * lr 

                    if args.print_update:
                        print("Grad norm:", torch.mean(torch.norm(torch.stack(grads_[-10:]))))
                    if torch.mean(torch.norm(torch.stack(grads_[-10:]))) < 0.001:
                        break
                    else:
                        count += 1
                acount += count
            counts.append(acount / len(initial_w))
            if args.visualize:
                ws = tree.map_structure(lambda x: float(x), ws)
                # print(ws)
                ws = np.array(ws)
                
                losses = tree.map_structure(lambda x: x.detach().cpu().numpy(), losses)
                plt.figure(figsize=(4, 4))
                for i in range(len(ws) - 1):
                    plt.arrow(ws[i, 0], ws[i, 1], ws[i + 1, 0] - ws[i, 0], ws[i+1, 1]-ws[i, 1], shape='full', color='b', lw=1, head_length=0.01, head_width=0.01)
                plt.title(f"{args.type},S={len(ws)}")
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_path.png", bbox_inches="tight")
                plt.close()
                plt.figure(figsize=(4, 4))
                
                plt.plot(list(range(len(losses) // 2)), losses[::2])
                plt.plot(list(range(len(losses) // 2)), losses[1::2])

                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_loss.png", bbox_inches="tight")
                plt.close()

                updates = torch.stack(updates).cpu().detach().numpy()
                grads_ = torch.stack(grads_).cpu().detach().numpy()
                Ags = torch.stack(Ags).cpu().detach().numpy()
                Sgs = torch.stack(Sgs).cpu().detach().numpy()
                plt.figure(figsize=(4, 4))
                plt.plot(range(updates.shape[0]), updates[:, 0,0], label='P1')
                plt.plot(range(updates.shape[0]), updates[:, 1,0], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_update_grad.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.plot(range(updates.shape[0]), updates[:, 0, 1], label='P1')
                plt.plot(range(updates.shape[0]), updates[:, 1, 1], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_update_A.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.plot(range(updates.shape[0]), updates[:, 0, 2], label='P1')
                plt.plot(range(updates.shape[0]), updates[:, 1, 2], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_update_S.png", bbox_inches="tight")
                plt.close()
                
                plt.figure(figsize=(4, 4))
                plt.plot(range(grads_.shape[0]), grads_[:, 0], label='P1')
                plt.plot(range(grads_.shape[0]), grads_[:, 1], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_grads.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.plot(range(Ags.shape[0]), Ags[:, 0], label='P1')
                plt.plot(range(Ags.shape[0]), Ags[:, 1], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_ags.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.plot(range(Sgs.shape[0]), Sgs[:, 0], label='P1')
                plt.plot(range(Sgs.shape[0]), Sgs[:, 1], label='P2')
                plt.legend()
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_sgs.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.plot(range(Sgs.shape[0]), np.sqrt(grads_[:, 0] ** 2 + grads_[:, 1] ** 2).reshape(-1))
                plt.savefig(f"{args.type}_batch/{idx}_{args.type}_grads_norm.png", bbox_inches="tight")
                plt.close()
    
    print('\t'.join(list(map(str, counts))))
if __name__ == "__main__":
    main()