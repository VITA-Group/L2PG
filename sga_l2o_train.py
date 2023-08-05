import abc
from operator import gt
import torch
import argparse
import numpy as np
from losses import *
from utils import generate_game_sample, load_games_list, construct_obs, init_stats, init_weight, detach, random_unit
from torch import nn
from networks import RNNOptimizer
import tree
import wandb
import copy

def slow_ema_update(slow_optimizer, optimizer, beta):
    for sp, p in zip(slow_optimizer.parameters(), optimizer.parameters()):
        sp.data = sp.data * beta + p.data * (1 - beta)

def main(args):

    wandb.init(project="l2o_game", name=args.wandb_name)
    wandb.config.update(args)
    torch.manual_seed(args.seed)

    cl = [50, 100, 200, 500, 1000]
    formula = args.formula.split(',')
    levels = args.feat_level.split(',')
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, args.epochs // 3)
    
    eval_game_list = load_games_list(args.eval_game_list, args.n_player)
    best_eval_result = 1000
    best_slow_eval_result = 1000
    best_soft_eval_result = 1000
    total_step = 0

    if args.cl:
        args.inner_iterations = cl[0]

    if args.use_slow_optimizer:
        slow_optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        slow_meta_optimizer = torch.optim.Adam(slow_optimizer.parameters(), lr=1e-3)
        slow_scheduler = torch.optim.lr_scheduler.StepLR(slow_meta_optimizer, args.epochs // 3)
    initialized = False
    for epoch in range(args.epochs):
        backup_weights = copy.deepcopy(optimizer.state_dict())
        if args.batch_size == 1:
            game_coef = generate_game_sample(args, epoch)
            loss, _ = loss_quadratic(1, *list(game_coef))
            epoch_w = init_weight(args.init_mode)
            for w in epoch_w:
                w.requires_grad = True
                w.retain_grad()
        else:
            game_coefs = []
            losses = []
            epoch_ws = []
            for _ in range(args.batch_size):
                game_coef = generate_game_sample(args, epoch)
                loss, _ = loss_quadratic(1, *list(game_coef))
                epoch_w = init_weight(args.init_mode)
                for w in epoch_w:
                    w.requires_grad = True
                    w.retain_grad()
                game_coefs.append(game_coef)
                losses.append(loss)
                epoch_ws.append(epoch_w)
        hiddens = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
        cells = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
        meta_loss = 0

        if (not initialized) and (epoch >= args.epochs * args.slow_optimizer_start) and (args.use_slow_optimizer):
            slow_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            initialized = True
        print(f"init location for fast: {epoch_w}")
        if args.use_slow_optimizer and initialized:
            slow_hiddens = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
            slow_cells = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
            if args.batch_size == 1:
                slow_epoch_w = init_weight()
                for w, sw in zip(epoch_w, slow_epoch_w):
                    sw.data.copy_(w.data)
                    sw.requires_grad = True
                    sw.retain_grad()
            else:
                slow_epoch_ws = []
                for _ in range(args.batch_size):
                    slow_epoch_w = init_weight()
                    for w, sw in zip(epoch_ws[_], slow_epoch_w):
                        sw.data.copy_(w.data)
                        sw.requires_grad = True
                        sw.retain_grad()
                    slow_epoch_ws.append(slow_epoch_w)
        elif args.self_soft_update:
            soft_params = copy.deepcopy(optimizer.state_dict())
        if args.batch_size == 1:
            init_grads = grad(loss, epoch_w).detach().clone()
        else:
            init_grads = [grad(loss, epoch_w).detach().clone() for epoch_w in epoch_ws]
        for iterations in range(args.inner_iterations):
            if args.batch_size == 1:
                grads = grad(loss, epoch_w) # (np * na) x 1
                S, A = decompose(grad(loss, epoch_w), epoch_w) # (np * na) x (np * na) 
                Ag = torch.transpose(A, 0, 1) @ grads
                Sg = S @ grads
            else:
                separate_grads = [grad(loss, epoch_w) for loss, epoch_w in zip(losses, epoch_ws)] # (np * na) x 1
                SAs = [decompose(grad(loss, epoch_w), epoch_w) for loss, epoch_w in zip(losses, epoch_ws)]
                Ag = [torch.transpose(A, 0, 1) @ grad for (S, A), grad in zip(SAs, separate_grads)]
                Sg = [S @ grad for (S, A), grad in zip(SAs, separate_grads)]
                grads = torch.cat(separate_grads) # (b * np * na) x 1
                Ag = torch.cat(Ag)
                Sg = torch.cat(Sg)
                
            obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)]           
            # obs = [grads.view(-1, 1)]            
            obs = torch.cat(obs, 1)
            if iterations == 0:
                stats = init_stats(obs, feat_levels=levels)

            obs, stats = construct_obs(obs, levels, stats, iterations)
            obs_overflow = False
            for i in range(obs.shape[1]):
                if torch.norm(obs[:, i]) > 1e9 or torch.isnan(torch.norm(obs[:, i])):
                    obs_overflow = True
                    break
                wandb.log({"obs_" + str(i): torch.norm(obs[:, i])}, step=total_step)
            if obs_overflow:
                break
            new_hs = []
            new_cs = []
            
            if args.reg_1:
                if args.batch_size == 1:
                    H = torch.sum(grads ** 2) / 2
                    dh = []
                    for g, w in zip(grads, epoch_w):
                        dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                    dh = torch.stack(dh)
                    loss_reg_1 = (grads.view(1,-1) @ dh.view(-1,1)).sum()
                    wandb.log({"step_loss_reg_1": loss_reg_1}, step=total_step)
                else:
                    Hs = [torch.sum(grad_ ** 2) / 2 for grad_ in separate_grads] 
                    losses_reg_1 = []
                    for g, w, H in zip(separate_grads, epoch_ws, Hs):
                        dh = []
                        for w_ in w:
                            dh.append(torch.autograd.grad(H, w_, retain_graph=True)[0])
                        dh = torch.stack(dh)
                        loss_reg_1 = (g.view(1,-1) @ dh.view(-1,1)).sum()
                        if loss_reg_1 < 0:
                            losses_reg_1.append(loss_reg_1)
                    if len(losses_reg_1) > 0:
                        loss_reg_1 = torch.mean(torch.tensor(losses_reg_1))
                    else:
                        loss_reg_1 = 0
                    
                    wandb.log({"step_loss_reg_1": loss_reg_1}, step=total_step)
            elif args.reg_2:
                if args.batch_size == 1:
                    H = torch.sum(grads ** 2) / 2
                    dh = []
                    for g, w in zip(grads, epoch_w):
                        dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                    dh = torch.stack(dh)
                    reg_2 = torch.sign((grads.view(1,-1) @ dh.view(-1,1)).sum())
                else:
                    Hs = [torch.sum(grad_ ** 2) / 2 for grad_ in separate_grads] 
                    reg_2s = []
                    for g, w, H in zip(separate_grads, epoch_ws, Hs):
                        dh = []
                        for w_ in w:
                            dh.append(torch.autograd.grad(H, w_, retain_graph=True)[0])
                        dh = torch.stack(dh)
                        reg_2 = (g.view(1,-1) @ dh.view(-1,1)).sum()
                        reg_2s.append(reg_2)
                    reg_2s = torch.sign(torch.stack(reg_2s))
                    
            update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
            # print(iterations, epoch_w, update)
            if args.batch_size == 1:
                epoch_w[0] = epoch_w[0] - (update[0,0] * grads[0] - update[0,1] * Ag[0] - update[0,2] * Sg[0]) * scale[0]
                epoch_w[1] = epoch_w[1] - (update[1,0] * grads[1] - update[1,1] * Ag[1] - update[1,2] * Sg[1]) * scale[1]
            else:
                for idx, epoch_w in enumerate(epoch_ws):
                    for j in range(args.n_player):
                        para_idx = j + args.n_player * idx
                        epoch_w[j] = epoch_w[j] - (update[para_idx, 0] * grads[para_idx] - update[para_idx, 1] * Ag[para_idx] - update[para_idx, 2] * Sg[para_idx]) * scale[0]
                        wandb.log({f"grad_{para_idx}": grads[para_idx],
                                   f"Ag_{para_idx}": Ag[para_idx],
                                   f"Sg_{para_idx}": Sg[para_idx],
                                   f"update0_{para_idx}": update[para_idx, 0],
                                   f"update1_{para_idx}": update[para_idx, 1],
                                   f"update2_{para_idx}": update[para_idx, 2]}, step=total_step)
                    # epoch_w[1] = epoch_w[1] - (update[1 + args.n_player * idx, 0] * grads[1] - update[1,1] * Ag[1] - update[1,2] * Sg[1]) * scale[1]
            new_hs.append(new_h)
            new_cs.append(new_c)
            print(f"updated location for fast: {epoch_w}")
            hiddens = new_hs
            cells = new_cs
            if args.normalize_meta_loss:
                step_meta_loss = torch.sum(grads ** 2) / torch.sum(init_grads ** 2) 
            else:
                if not args.reg_2:
                    step_meta_loss = 1 / 2 * torch.sum(grads ** 2) / args.batch_size
                else:
                    step_meta_loss = torch.sum((grads ** 2).view(args.batch_size, -1) * reg_2s.view(args.batch_size, -1), 1)
                    step_meta_loss = step_meta_loss[(step_meta_loss > -100) & (~torch.isnan(step_meta_loss))]
                    step_meta_loss = torch.mean(step_meta_loss) / 2
            total_step = total_step + 1
            wandb.log({"step_meta_loss": step_meta_loss}, step=total_step)
            meta_loss += step_meta_loss
            if args.reg_1:
                if loss_reg_1 < 0:
                    meta_loss -= loss_reg_1 * args.reg_coef
                        
            if (meta_loss > 1e5) or (meta_loss < -1e5):
                break
            elif torch.isnan(meta_loss):
                optimizer.load_state_dict(backup_weights)
                break
            elif (iterations + 1) % args.unroll_length == 0:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer.parameters(), 1)
                meta_optimizer.step()
                optimizer.zero_grad()
                if args.self_soft_update:
                    state_dict = optimizer.state_dict()
                    for key in soft_params:
                        soft_params[key] = 0.9 * soft_params[key] + 0.1 * state_dict[key]

                print(iterations + 1, f'meta loss: {meta_loss.item()}', f'current loss: {torch.sum(grads ** 2)}')
                wandb.log({"meta_loss": meta_loss}, step=total_step)
                meta_loss = 0
                hiddens = tree.map_structure(detach, hiddens)
                cells = tree.map_structure(detach, cells)
                if args.batch_size == 1:
                    epoch_w = tree.map_structure(detach, epoch_w)
                else:
                    epoch_ws = tree.map_structure(detach, epoch_ws)
                del Ag
                del Sg

            if args.use_slow_optimizer and initialized:
  
                if iterations == 0 or iterations % args.slow_optimizer_freq == 0:
                    if iterations > 0:
                        slow_meta_loss = 0
                        if args.batch_size == 1:
                            for sw, w in zip(slow_epoch_w, epoch_w):
                                slow_meta_loss += torch.sum((sw - w.data) ** 2) 
                        else:
                            for sws, ws in zip(slow_epoch_ws, epoch_ws):
                                for sw, w in zip(sws, ws):
                                    slow_meta_loss += torch.sum((sw - w.data) ** 2) / args.batch_size
                                
                        if slow_meta_loss < 1e5:
                            slow_meta_loss.backward()
                            torch.nn.utils.clip_grad_norm_(slow_optimizer.parameters(), 1)
                            slow_meta_optimizer.step()
                            slow_optimizer.zero_grad()
                            print(iterations + 1, f'slow meta loss: {slow_meta_loss.item()}')

                            if args.use_slow_ema:
                                slow_ema_update(slow_optimizer, optimizer, args.slow_ema)
                        else:
                            # print(slow_grads)
                            # print(slow_S, slow_A, slow_Ag, slow_Sg)
                            # print(slow_obs)
                            # assert False
                            print("Overflow.")
                            slow_optimizer.zero_grad()
                        wandb.log({"slow_meta_loss": slow_meta_loss}, step=total_step)
                        slow_meta_loss = 0
                        slow_hiddens = tree.map_structure(detach, slow_hiddens)
                        slow_cells = tree.map_structure(detach, slow_cells)
                        if args.batch_size == 1:
                            slow_epoch_w = tree.map_structure(detach, slow_epoch_w)
                        else:
                            slow_epoch_ws = tree.map_structure(detach, slow_epoch_ws)

                    if args.batch_size == 1:
                        slow_grads = grad(loss, slow_epoch_w) # (np * na) x 1
                        slow_S, slow_A = decompose(grad(loss, slow_epoch_w), slow_epoch_w) # (np * na) x (np * na) 
                        slow_Ag = torch.transpose(slow_A, 0, 1) @ slow_grads
                        slow_Sg = slow_S @ slow_grads
                    else:
                        slow_separate_grads = [grad(loss, slow_epoch_w) for loss, slow_epoch_w in zip(losses, slow_epoch_ws)] # (np * na) x 1
                        slow_SAs = [decompose(grad(loss, slow_epoch_w), slow_epoch_w) for loss, slow_epoch_w in zip(losses, slow_epoch_ws)]
                        slow_Ag = [torch.transpose(slow_A, 0, 1) @ slow_grad for (slow_S, slow_A), slow_grad in zip(slow_SAs, slow_separate_grads)]
                        slow_Sg = [slow_S @ slow_grad for (slow_S, slow_A), slow_grad in zip(slow_SAs, slow_separate_grads)]
                        slow_grads = torch.cat(slow_separate_grads) # (b * np * na) x 1
                        slow_Ag = torch.cat(slow_Ag)
                        slow_Sg = torch.cat(slow_Sg)

                    slow_obs = [slow_grads.view(-1, 1), slow_Ag.view(-1, 1), slow_Sg.view(-1, 1)]    
                    # slow_obs = [slow_grads.view(-1, 1)]
                    slow_obs = torch.cat(slow_obs, 1)
                    slow_stats = init_stats(slow_obs, feat_levels=levels)
                    slow_obs, slow_stats = construct_obs(slow_obs, levels, slow_stats, iterations // args.slow_optimizer_freq)  
  
                    slow_new_hs = []
                    slow_new_cs = []
                    
                    slow_update, slow_scale, slow_new_h, slow_new_c = slow_optimizer(slow_obs, slow_hiddens[0], slow_cells[0])
                    if args.batch_size == 1:
                        slow_epoch_w[0] = slow_epoch_w[0] - (slow_update[0,0] * slow_grads[0] - slow_update[0,1] * slow_Ag[0] - slow_update[0,2] * slow_Sg[0]) * slow_scale[0]
                        slow_epoch_w[1] = slow_epoch_w[1] - (slow_update[1,0] * slow_grads[1] - slow_update[1,1] * slow_Ag[1] - slow_update[1,2] * slow_Sg[1]) * slow_scale[1]
                    else:
                        for idx, slow_epoch_w in enumerate(slow_epoch_ws):
                            for j in range(args.n_player):
                                para_idx = j + args.n_player * idx
                                slow_epoch_w[j] = slow_epoch_w[j] - (slow_update[para_idx, 0] * slow_grads[para_idx] - slow_update[para_idx, 1] * slow_Ag[para_idx] - slow_update[para_idx, 2] * slow_Sg[para_idx]) * slow_scale[0]
                    print(f"updated location for slow: {slow_epoch_w}")
                    slow_new_hs.append(slow_new_h)
                    slow_new_cs.append(slow_new_c)
                    slow_hiddens = slow_new_hs
                    slow_cells = slow_new_cs
        
    
        if (epoch + 1) % 5 == 0:
            eval_result = evaluate(optimizer, eval_game_list, formula, levels, args)
            mean_eval_result = np.mean(eval_result)
            if args.use_slow_optimizer and initialized:
                slow_eval_result = np.mean(evaluate(slow_optimizer, eval_game_list, formula, levels, args, slow=True))
                if slow_eval_result < best_slow_eval_result:
                    torch.save({"state_dict": slow_optimizer.state_dict(), "eval_results": mean_eval_result}, "slow_" + args.output_name)
                    best_slow_eval_result = slow_eval_result
                wandb.log({'slow_mean_eval_result': np.mean(slow_eval_result)}, step=total_step)
            if args.self_soft_update:
                original_state_dict = copy.deepcopy(optimizer.state_dict())
                optimizer.load_state_dict(soft_params)
                soft_eval_result = np.mean(evaluate(optimizer, eval_game_list, formula, levels, args))
                wandb.log({'self_soft_mean_eval_result': soft_eval_result}, step=total_step)
                optimizer.load_state_dict(original_state_dict)
                if soft_eval_result < best_soft_eval_result:
                    torch.save({"state_dict": soft_params, "eval_results": soft_eval_result}, "soft_" + args.output_name)
            if mean_eval_result < best_eval_result:
                best_eval_result = mean_eval_result
                torch.save({"state_dict": optimizer.state_dict(), "eval_results": mean_eval_result}, args.output_name)
                
            elif args.cl:
                try:
                    args.inner_iterations = cl[cl.index(args.inner_iterations) + 1]
                except IndexError:
                    pass
            wandb.log({'mean_eval_result': mean_eval_result}, step=total_step)
        scheduler.step()
        if args.use_slow_optimizer:
            slow_scheduler.step()

        

def evaluate(net, game_list, formula, levels, args, slow=False):
    counts = []
    for idx, loss_line in enumerate(game_list):
        optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        optimizer.load_state_dict(net.state_dict())
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
                
                grads = grad(loss, players_w).cuda()
                S, A = decompose(grads, players_w) # (np * na) x (np * na) 
                Ate = torch.transpose(A, 0, 1) @ grads
                Ss = S @ grads
                obs = [grads.view(-1, 1), Ate.view(-1, 1), Ss.view(-1, 1)]
                # obs = [grads.view(-1, 1)]
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
                if not slow:
                    players_w[0] = players_w[0] - (grads[0] * update[0,0] - Ate[0] * update[0, 1] - Ss[0] * update[0, 2]) * scale[0]
                    players_w[1] = players_w[1] - (grads[1] * update[1,0] - Ate[1] * update[1, 1] - Ss[1] * update[1, 2]) * scale[1]
                else:
                    players_w[0] = players_w[0] - (grads[0] * update[0,0] - Ate[0] * update[0, 1] - Ss[0] * update[0, 2]) * scale[0] 
                    players_w[1] = players_w[1] - (grads[1] * update[1,0] - Ate[1] * update[1, 1] - Ss[1] * update[1, 2]) * scale[1] 
                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs
                
                if torch.mean(torch.norm(torch.stack(grads_[-10:]), dim=1)) < 0.001:
                    break
                else:
                    count += 1
            acount += count

        counts.append(acount / len(initial_w))

    return counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_player', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=20)
    parser.add_argument('--n_action', type=int, default=1)
    parser.add_argument('--reg_1', action='store_true')
    parser.add_argument('--reg_2', action='store_true')
    parser.add_argument('--reg_coef', type=float, default=10)
    parser.add_argument('--formula', type=str, default='grad,S,A')
    parser.add_argument('--learnable_scale', action='store_true')
    #### Game Type #### 
    parser.add_argument('--stable', action='store_true')
    parser.add_argument('--stable-saddle', action='store_true')
    parser.add_argument('--game-distribution', type=str, default='gaussian', choices=['gaussian', 'uniform', 'negative-uniform'])
    parser.add_argument('--output-name', type=str, default='optimizer.pkl')
    parser.add_argument('--wandb-name', type=str, default='meta-train')
    parser.add_argument('--inner-iterations', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--feat_level', type=str, default="o,m,0.9")
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--eval-game-list', type=str, default='stable_game_list_uniform.txt')
    parser.add_argument('--cl', action="store_true")
    parser.add_argument('--learnable-loss', action='store_true', help='enable learnable loss or not')

    parser.add_argument('--use-slow-optimizer', action="store_true", help='enable slow optimizer')
    parser.add_argument('--use-slow-ema', action="store_true", help='enable slow ema')
    parser.add_argument('--slow-ema', type=float, default=0.95)
    parser.add_argument('--slow-optimizer-start', type=float, default=0.1)

    parser.add_argument('--normalize-meta-loss', action="store_true", help='enable slow ema')

    parser.add_argument('--slow-optimizer-freq', type=int, default=5)
    parser.add_argument('--loss-type', type=str, default='mse', choices=('mse', 'cosine'))
    parser.add_argument('--init-mode', type=str, default='unit', choices=('unit', 'ball'))
    parser.add_argument('--no-tanh', action='store_true')
    parser.add_argument('--data-cl', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--self-soft-update', action='store_true')

    args = parser.parse_args()
    main(args)