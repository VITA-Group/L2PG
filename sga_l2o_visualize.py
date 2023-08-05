import abc
import torch
import argparse
import numpy as np
from losses import *
import torch.nn as nn
from utils import generate_game_sample, load_games_list, construct_obs, init_stats, init_weight, detach, random_unit
from networks import RNNOptimizer
import tree
import wandb 

def main(args):


    torch.manual_seed(args.seed)

    cl = [50, 100, 200, 500, 1000]
    formula = args.formula.split(',')
    levels = args.feat_level.split(',')
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), learnable_scale=args.learnable_scale).cuda()
    try:
        optimizer.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    except:
        optimizer.load_state_dict(torch.load(args.checkpoint))

    eval_game_list = load_games_list(args.game_list, args.n_player)
    best_eval_result = 1000
    total_step = 0
    
    if args.cl:
        args.inner_iterations = cl[0]
    if args.learnable_loss:
        from learnable_losses import LearnableLoss
        learned_loss = LearnableLoss(2, zdim=0).cuda()

    
    counts, grads_, Ags, Sgs, ws, updates = evaluate(optimizer, eval_game_list, formula, levels, args, slow=args.use_slow_optimizer)
    ws = tree.map_structure(lambda x: float(x), ws)
    # print(ws)
    ws = np.array(ws)
    import matplotlib.pyplot as plt

    updates = torch.stack(updates).cpu().detach().numpy()
    grads_ = torch.stack(grads_).cpu().detach().numpy()
    Ags = torch.stack(Ags).cpu().detach().numpy()
    Sgs = torch.stack(Sgs).cpu().detach().numpy()

    output_data = {'P1_update_0': updates[:,0,0], 'P2_update_0': updates[:,1,0],'P1_update_1': updates[:,0,1],'P2_update_1': updates[:,1,1], 'P1_update_2': updates[:,0,2],'P2_update_2': updates[:,1,2], 'P1_grads': grads_[:, 0], 'P2_grads': grads_[:, 1], 'P1_Ag': Ags[:, 0], 'P2_Ag': Ags[:, 1], 'P1_Sg': Sgs[:, 0], 'P2_Sg': Sgs[:, 1]}
    import pandas as pd
    pd.DataFrame(output_data).to_csv(f'{args.output_name}.csv')
    print(updates.shape)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.plot(range(updates.shape[0]), updates[:, 0,0], label='P1')
    plt.plot(range(updates.shape[0]), updates[:, 1,0], label='P2')
    plt.legend()


    plt.subplot(2, 3, 2)
    plt.plot(range(updates.shape[0]), updates[:, 0, 1], label='P1')
    plt.plot(range(updates.shape[0]), updates[:, 1, 1], label='P2')
    plt.legend()


    plt.subplot(2, 3, 3)
    plt.plot(range(updates.shape[0]), updates[:, 0, 2], label='P1')
    plt.plot(range(updates.shape[0]), updates[:, 1, 2], label='P2')
    plt.legend()

    
    plt.subplot(2, 3, 4)
    plt.plot(range(grads_.shape[0]), grads_[:, 0], label='P1')
    plt.plot(range(grads_.shape[0]), grads_[:, 1], label='P2')
    plt.legend()


    plt.subplot(2, 3, 5)
    plt.plot(range(Ags.shape[0]), Ags[:, 0], label='P1')
    plt.plot(range(Ags.shape[0]), Ags[:, 1], label='P2')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(range(Sgs.shape[0]), Sgs[:, 0], label='P1')
    plt.plot(range(Sgs.shape[0]), Sgs[:, 1], label='P2')
    plt.legend()
    plt.savefig(f"{args.output_name}_{args.game_idx}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    print(len(ws))
    for i in range(len(ws) - 1):
        plt.arrow(ws[i, 0], ws[i, 1], ws[i + 1, 0] - ws[i, 0], ws[i+1, 1]-ws[i, 1], shape='full', color='b', lw=1, head_length=0.01, head_width=0.01)
    plt.title(f"L2O,S={len(ws)}")
    plt.savefig(f"{args.output_name}_path_{args.game_idx}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    print(len(ws))
    for i in range(len(ws) - 1):
        plt.arrow(ws[i, 0], ws[i, 1], ws[i + 1, 0] - ws[i, 0], ws[i+1, 1]-ws[i, 1], shape='full', color='b', lw=1, head_length=0.01, head_width=0.01)
    plt.title(f"L2O,S={len(ws)}")
    plt.savefig(f"{args.output_name}_path_{args.game_idx}.png", bbox_inches="tight")
    plt.close()

    pd.DataFrame({'p1': ws[:, 0], 'p2': ws[:, 1]}).to_csv(f'{args.output_name}_path.csv')
def evaluate(net, game_list, formula, levels, args, slow=False):
    counts = []
    beta1 = 0.9
    beta2 = 0.99
    for idx, loss_line in enumerate(game_list):
        if idx != args.game_idx: continue
        optimizer = RNNOptimizer(True, args.n_hidden, 10, False, learnable_scale=args.learnable_scale, n_features=len(formula) * (len(levels))).cuda()
        optimizer.load_state_dict(net.state_dict())
        lrs = []
        ws = []
        updates = []
        grads_ = []
        Ags = []
        Sgs = []
        loss, _ = loss_quadratic(1, *list(loss_line))
        ws = []
        initial_w = [random_unit(2**int(_)) for _ in range(1, 2)]
        acount = 0
        trajectory = []
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
                
                grads = grad(loss, players_w) # (np * na) x 1
                S, A = decompose(grad(loss, players_w), players_w) # (np * na) x (np * na) 
                Ag = torch.transpose(A, 0, 1) @ grads
                Sg = S @ grads
                obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)]            
                obs = torch.cat(obs, 1)
                if count == 0:
                    stats = init_stats(obs, feat_levels=levels)
                obs, stats = construct_obs(obs, levels, stats, count)
                new_hs = []
                new_cs = []                    
                with torch.no_grad():
                    update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
                # print(update)
                updates.append(update)
                grads_.append(grads.view(-1) * update[:, 0].view(-1))
                Ags.append(Ag.view(-1) * update[:, 1].view(-1))
                Sgs.append(Sg.view(-1) * update[:, 2].view(-1))
                losses.extend(loss(players_w))
                ws.append(list(players_w))
                with torch.no_grad():
                    players_w[0] = players_w[0] - (grads[0] * update[0,0] - Ag[0] * update[0, 1] - Sg[0] * update[0, 2]) * scale[0] 
                    players_w[1] = players_w[1] - (grads[1] * update[1,0] - Ag[1] * update[1, 1] - Sg[1] * update[1, 2]) * scale[1] 
                players_w[0].requires_grad = True
                players_w[1].requires_grad = True
                players_w[0].retain_grad()
                players_w[1].retain_grad()

                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs
                # print(torch.norm(torch.stack(grads_[-10:]), dim=1))

                if torch.mean(torch.norm(torch.stack(grads_[-10:]), dim=1)) < 0.001:
                    break
                else:
                    count += 1
                # print(ws)
            acount += count

        counts.append(acount / len(initial_w))
        break
    return counts, grads_, Ags, Sgs, ws, updates

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_player', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=20)
    parser.add_argument('--n_action', type=int, default=1)
    parser.add_argument('--reg_1', action='store_true')
    parser.add_argument('--formula', type=str, default='grad,S,A')
    parser.add_argument('--learnable_scale', action='store_true')
    #### Game Type #### 
    parser.add_argument('--stable', action='store_true')
    parser.add_argument('--stable-saddle', action='store_true')
    parser.add_argument('--game-distribution', type=str, default='gaussian', choices=['gaussian', 'uniform', 'negative-uniform'])
    parser.add_argument('--inner-iterations', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--feat_level', type=str, default="o,m,0.9")
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--wandb-name', type=str, default="meta-train")
    parser.add_argument('--eval-game-list', type=str, default='stable_game_list_uniform.txt')
    parser.add_argument('--cl', action="store_true")
    parser.add_argument('--learnable-loss', action='store_true', help='enable learnable loss or not')
    parser.add_argument("--game_list", type=str)
    parser.add_argument('--use-slow-optimizer', action="store_true", help='enable slow optimizer')
    parser.add_argument('--use-slow-ema', action="store_true", help='enable slow ema')

    parser.add_argument('--slow-optimizer-freq', type=int, default=5)
    parser.add_argument('--loss-type', type=str, default='mse', choices=('mse', 'cosine'))

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--slow-checkpoint', type=str)
    parser.add_argument('--game-idx', type=int, default=0)
    parser.add_argument('--output-name', type=str)

    args = parser.parse_args()
    main(args)