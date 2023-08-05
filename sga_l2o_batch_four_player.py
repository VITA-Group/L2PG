import abc
import torch
import argparse
import numpy as np
from losses import *
from networks import RNNOptimizer
import matplotlib.pyplot as plt
import tree
from utils import generate_game_sample, load_games_list, construct_obs, init_stats, init_weight, detach, random_unit
parser = argparse.ArgumentParser()
parser.add_argument("--n_player", type=int, default=2)
parser.add_argument("--n_action", type=int, default=1)
parser.add_argument("--n_hidden", type=int, default=16)
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--random-lr', action="store_true")
parser.add_argument('--checkpoint', type=str)
parser.add_argument("--eval_game_list", type=str)
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--print_update", action="store_true")
parser.add_argument("--no-tanh", action="store_true")
parser.add_argument("--learnable_scale", action="store_true")
parser.add_argument("--formula", type=str, default='grad,S,A')
parser.add_argument('--feat_level', type=str, default="o,m,0.9")
args = parser.parse_args()
from sga_l2o_train_four_player import evaluate
from tqdm import tqdm
def main():
    torch.manual_seed(args.seed)
    # eval_game_list = [-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1]# load_games_list(args.eval_game_list, args.n_player)
    eval_game_list = list([0.6, 1])
    print(eval_game_list)
    counts = []
    formula = args.formula.split(",")
    levels = args.feat_level.split(',')
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
    try:
        optimizer.load_state_dict(torch.load(args.checkpoint))
    except:
        optimizer.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    eval_result = evaluate(optimizer, eval_game_list, formula, levels, args)
    print(eval_result)
    print(np.mean(eval_result))
    '''
    for idx, loss_line in enumerate(eval_game_list):
        optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        try:
            optimizer.load_state_dict(torch.load(args.checkpoint))
        except:
            optimizer.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        optimizer.eval()
        lrs = []
        ws = []
        updates = []
        grads_ = []
        Ags = []
        Sgs = []
        loss, _ = loss_quadratic(1, *list(loss_line))
        ws = []
        initial_w = [random_unit(2**int(_)) for _ in range(1, 5)]
        acount = 0
        for wi in initial_w:
            players_w = wi
            ws.append(list(wi))
            losses = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
                w.cuda()

            hiddens = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            cells = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            count = 0

            while count < 1000:
                
                grads = torch.stack(grad(loss, players_w)).cuda()
                S, A = decompose(grads, players_w) # (np * na) x (np * na) 
                Ate = torch.transpose(A, 0, 1) @ grads
                Ss = S @ grads
                obs = [grads.view(-1, 1), Ate.view(-1, 1), Ss.view(-1, 1)]
                obs = torch.cat(obs, 1)
                if count == 0:
                    stats = init_stats(obs, feat_levels=levels)
                obs, stats = construct_obs(obs, levels, stats, count)
                new_hs = []
                new_cs = []                    
                with torch.no_grad():
                    update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])

                updates.append(update)
                grads_.append(grads.view(-1) * update[:, 0].view(-1))
                Ags.append(Ate.view(-1) * update[:, 1].view(-1))
                Sgs.append(Ss.view(-1) * update[:, 2].view(-1))
                losses.extend(loss(players_w))
                ws.append(list(players_w))
        
                players_w[0] = players_w[0] - (grads[0] * update[0,0] - Ate[0] * update[0, 1] - Ss[0] * update[0, 2]) * scale[0] 
                players_w[1] = players_w[1] - (grads[1] * update[1,0] - Ate[1] * update[1, 1] - Ss[1] * update[1, 2]) * scale[1] 
                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs
                
                if torch.mean(torch.norm(torch.stack(grads_[-10:]))) < 0.001:
                    break
                else:
                    count += 1
            acount += count

        counts.append(acount / len(initial_w))  
    print(counts)
    print(np.mean(counts))
    '''
if __name__ == "__main__":
    main()