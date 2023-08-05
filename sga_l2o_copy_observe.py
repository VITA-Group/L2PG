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
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), learnable_scale=args.learnable_scale).cuda()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, args.epochs // 3)
    
    eval_game_list = load_games_list(args)
    best_eval_result = 1000
    total_step = 0

    if args.cl:
        args.inner_iterations = cl[0]
    if args.learnable_loss:
        from learnable_losses import LearnableLoss
        learned_loss = LearnableLoss(2, zdim=0).cuda()
        loss_optimizer = torch.optim.Adam(learned_loss.parameters(), 0.05)

    if args.use_slow_optimizer:
        slow_optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), learnable_scale=args.learnable_scale).cuda()
        slow_meta_optimizer = torch.optim.Adam(slow_optimizer.parameters(), lr=1e-3)
        slow_scheduler = torch.optim.lr_scheduler.StepLR(slow_meta_optimizer, args.epochs // 3)
    for epoch in range(args.epochs):
        game_coef = generate_game_sample(args)
        loss, _ = loss_quadratic(1, *list(game_coef))
        epoch_w = init_weight()
        for w in epoch_w:
            w.requires_grad = True
            w.retain_grad()
        hiddens = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
        cells = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
        meta_loss = 0

        if args.use_slow_optimizer:
            slow_hiddens = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            slow_cells = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            slow_epoch_w = init_weight()
            for w, sw in zip(epoch_w, slow_epoch_w):
                sw.data.copy_(w.data)
                sw.requires_grad = True
                sw.retain_grad()
        for iterations in range(args.inner_iterations):
            
            grads = torch.cat(grad(loss, epoch_w), 0) # (np * na) x 1
            S, A = decompose(torch.stack(grad(loss, epoch_w)), epoch_w) # (np * na) x (np * na) 
            Ag = torch.transpose(A, 0, 1) @ grads
            Sg = S @ grads
            obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)]            
            obs = torch.cat(obs, 1)
            if iterations == 0:
                stats = init_stats(obs, feat_levels=levels)

            obs, stats = construct_obs(obs, levels, stats, iterations)

            for i in range(obs.shape[1]):
                wandb.log({"obs_" + str(i): torch.norm(obs[:, i])}, step=total_step)

            new_hs = []
            new_cs = []
            
            if args.reg_1:
                H = torch.sum(grads ** 2) / 2
                dh = []
                for g, w in zip(grads, epoch_w):
                    dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                dh = torch.stack(dh)
                loss_reg_1 = (grads.view(1,-1) @ dh.view(-1,1)).sum()

            update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
            # print(iterations, epoch_w, update)
            epoch_w[0] = epoch_w[0] - (update[0,0] * grads[0] - update[0,1] * Ag[0] - update[0,2] * Sg[0]) * scale[0]
            epoch_w[1] = epoch_w[1] - (update[1,0] * grads[1] - update[1,1] * Ag[1] - update[1,2] * Sg[1]) * scale[1]
            new_hs.append(new_h)
            new_cs.append(new_c)
            hiddens = new_hs
            cells = new_cs

            step_meta_loss = 1 / 2 * torch.sum(grads ** 2)
            total_step = total_step + 1
            wandb.log({"step_meta_loss": step_meta_loss}, step=total_step)
            meta_loss += 1 / 2 * torch.sum(grads ** 2)
            if args.reg_1:
                if loss_reg_1 < 0:
                    meta_loss -= loss_reg_1 * 10
                    print(loss_reg_1)
                        
            if meta_loss > 1e6:
                break
            elif (iterations + 1) % args.unroll_length == 0:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer.parameters(), 1)
                meta_optimizer.step()
                optimizer.zero_grad()
                print(iterations + 1, f'meta loss: {meta_loss.item()}', f'current loss: {torch.sum(grads ** 2)}')
                wandb.log({"meta_loss": meta_loss}, step=total_step)
                meta_loss = 0
                hiddens = tree.map_structure(detach, hiddens)
                cells = tree.map_structure(detach, cells)
                epoch_w = tree.map_structure(detach, epoch_w)
                del Ag
                del Sg

            if args.use_slow_optimizer:
  
                if iterations == 0 or iterations % args.slow_optimizer_freq == 0:
                    if iterations > 0:
                        slow_meta_loss = 0
                        for sw, w in zip(slow_epoch_w, epoch_w):
                            slow_meta_loss += torch.sum((sw - w.data) ** 2)
                        if slow_meta_loss < 1e5:
                            slow_meta_loss.backward()
                            torch.nn.utils.clip_grad_norm_(slow_optimizer.parameters(), 1)
                            slow_meta_optimizer.step()
                            slow_optimizer.zero_grad()
                            print(iterations + 1, f'slow meta loss: {meta_loss.item()}')

                            if args.use_slow_ema:
                                slow_ema_update(slow_optimizer, optimizer, 0.95)

                        else:
                            print("Overflow.")
                            slow_optimizer.zero_grad()
                        wandb.log({"slow_meta_loss": slow_meta_loss}, step=total_step)
                        slow_meta_loss = 0
                        slow_hiddens = tree.map_structure(detach, slow_hiddens)
                        slow_cells = tree.map_structure(detach, slow_cells)
                        slow_epoch_w = tree.map_structure(detach, slow_epoch_w)

                    slow_grads = torch.cat(grad(loss, slow_epoch_w), 0) # (np * na) x 1
                    slow_S, slow_A = decompose(torch.stack(grad(loss, slow_epoch_w)), slow_epoch_w) # (np * na) x (np * na) 
                    slow_Ag = torch.transpose(slow_A, 0, 1) @ slow_grads
                    slow_Sg = slow_S @ slow_grads
                    slow_obs = [slow_grads.view(-1, 1), slow_Ag.view(-1, 1), slow_Sg.view(-1, 1)]            
                    slow_obs = torch.cat(slow_obs, 1)
                    slow_stats = init_stats(slow_obs, feat_levels=levels)
                    slow_obs, slow_stats = construct_obs(slow_obs, levels, slow_stats, iterations // args.slow_optimizer_freq)  
  
                    slow_new_hs = []
                    slow_new_cs = []
                    
                    slow_update, slow_scale, slow_new_h, slow_new_c = slow_optimizer(slow_obs, slow_hiddens[0], slow_cells[0])
                    slow_epoch_w[0] = slow_epoch_w[0] - (slow_update[0,0] * slow_grads[0] - slow_update[0,1] * slow_Ag[0] - slow_update[0,2] * slow_Sg[0]) * slow_scale[0] * args.slow_optimizer_freq
                    slow_epoch_w[1] = slow_epoch_w[1] - (slow_update[1,0] * slow_grads[1] - slow_update[1,1] * slow_Ag[1] - slow_update[1,2] * slow_Sg[1]) * slow_scale[1] * args.slow_optimizer_freq

                    slow_new_hs.append(slow_new_h)
                    slow_new_cs.append(slow_new_c)
                    slow_hiddens = slow_new_hs
                    slow_cells = slow_new_cs
                    
        if args.learnable_loss:
            epoch_w = init_weight()
            for w in epoch_w:
                w.requires_grad = True
                w.retain_grad()
            hiddens = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            cells = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
            meta_loss = 0
            for iterations in range(args.inner_iterations):
                
                grads = torch.cat(grad(learned_loss, epoch_w), 0) # (np * na) x 1
                S, A = decompose(torch.stack(grad(learned_loss, epoch_w)), epoch_w) # (np * na) x (np * na) 
                Ag = torch.transpose(A, 0, 1) @ grads
                Sg = S @ grads
                obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)]            
                obs = torch.cat(obs, 1)
                if iterations == 0:
                    stats = init_stats(obs, feat_levels=levels)

                obs, stats = construct_obs(obs, levels, stats, iterations)

                for i in range(obs.shape[1]):
                    wandb.log({"obs_" + str(i): torch.norm(obs[:, i])}, step=total_step)

                new_hs = []
                new_cs = []
                
                if args.reg_1:
                    H = torch.sum(grads ** 2) / 2
                    dh = []
                    for g, w in zip(grads, epoch_w):
                        dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                    dh = torch.stack(dh)
                    loss_reg_1 = (grads.view(1,-1) @ dh.view(-1,1)).sum()

                update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
                # print(iterations, epoch_w, update)
                epoch_w[0] = epoch_w[0] - (update[0,0] * grads[0] - update[0,1] * Ag[0] - update[0,2] * Sg[0]) * scale[0]
                epoch_w[1] = epoch_w[1] - (update[1,0] * grads[1] - update[1,1] * Ag[1] - update[1,2] * Sg[1]) * scale[1]
                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs

                step_meta_loss = 1 / 2 * torch.sum(grads ** 2)
                total_step = total_step + 1
                wandb.log({"step_meta_loss": step_meta_loss}, step=total_step)
                meta_loss += 1 / 2 * torch.sum(grads ** 2)
                if args.reg_1:
                    if loss_reg_1 < 0:
                        meta_loss -= loss_reg_1 * 10
                        print(loss_reg_1)
            
                if meta_loss > 1e6:
                    break

                elif (iterations + 1) % args.unroll_length == 0:
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(optimizer.parameters(), 1)
                    meta_optimizer.step()
                    optimizer.zero_grad()
                    print(iterations + 1, f'meta loss: {meta_loss.item()}', f'current loss: {torch.sum(grads ** 2)}')
                    wandb.log({"meta_loss": meta_loss}, step=total_step)
                    meta_loss = 0
                    hiddens = tree.map_structure(detach, hiddens)
                    cells = tree.map_structure(detach, cells)
                    epoch_w = tree.map_structure(detach, epoch_w)
                    del Ag
                    del Sg
        


        if epoch % 5 == 0:
            eval_result = evaluate(optimizer, eval_game_list, formula, levels, args)
            mean_eval_result = np.mean(eval_result)
            if args.use_slow_optimizer:
                slow_eval_result = evaluate(slow_optimizer, eval_game_list, formula, levels, args, slow=True)
                torch.save({"state_dict": optimizer.state_dict(), "eval_results": mean_eval_result}, "slow_" + args.output_name)
                wandb.log({'slow_mean_eval_result': np.mean(slow_eval_result)}, step=total_step)
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
        slow_scheduler.step()


        if args.learnable_loss:
            avg_loss = train_loss(learned_loss, optimizer, loss_optimizer, levels, 10)
            torch.save(learned_loss.state_dict(), 'loss_quadratic.pkl')
            wandb.log({'learnable_loss_avg': avg_loss}, step=total_step)

        

def evaluate(net, game_list, formula, levels, args, slow=False):
    counts = []
    beta1 = 0.9
    beta2 = 0.99
    for idx, loss_line in enumerate(game_list):
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
                if slow:
                    obs, stats = construct_obs(obs, levels, stats, count // args.slow_optimizer_freq)
                else:
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
                    players_w[0] = players_w[0] - (grads[0] * update[0,0] - Ate[0] * update[0, 1] - Ss[0] * update[0, 2]) * scale[0] * args.slow_optimizer_freq
                    players_w[1] = players_w[1] - (grads[1] * update[1,0] - Ate[1] * update[1, 1] - Ss[1] * update[1, 2]) * scale[1] * args.slow_optimizer_freq
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

    return counts


def train_loss(loss, optimizer, loss_optimizer, levels, iterations=50):
    print("Training loss: ")
    avg_loss = []
    for i in range(iterations):
        meta_loss = 0
        loss_fn = loss
        epoch_w = init_weight()
        for w in epoch_w:
            w.requires_grad = True
            w.retain_grad()
        hiddens = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
        cells = [[torch.zeros(w.numel() * args.n_player, args.n_hidden).cuda()]]
        for step in range(10):
            grads = torch.stack(grad(loss, epoch_w)).cuda()
            S, A = decompose(grads, epoch_w) # (np * na) x (np * na) 
            Ate = torch.transpose(A, 0, 1) @ grads
            Ss = S @ grads
            obs = [grads.view(-1, 1), Ate.view(-1, 1), Ss.view(-1, 1)]
            obs = torch.cat(obs, 1)
            if step == 0:
                stats = init_stats(obs, feat_levels=levels)

            obs, stats = construct_obs(obs, levels, stats, step)
            new_hs = []
            new_cs = []
            update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
            epoch_w[0] = epoch_w[0] - grads[0] * update[0,0] - Ate[0] * update[0, 1] - Ss[0] * update[0, 2]
            epoch_w[1] = epoch_w[1] - grads[1] * update[1,0] - Ate[1] * update[1, 1] - Ss[1] * update[1, 2]
            new_hs.append(new_h)
            new_cs.append(new_c)
            hiddens = new_hs
            cells = new_cs
            meta_loss -= 0.001 / 2 * torch.sum(grads ** 2) # gradient ascent
            # print(meta_loss)
        
        print(f"iteration {i}, loss: {-meta_loss}")
        if not torch.isnan(meta_loss) and abs(meta_loss) < 1e8:
            
            meta_loss.backward()
            avg_loss.append(meta_loss.item())
            # for p in generator.parameters():
                # print(p.grad)
            nn.utils.clip_grad.clip_grad_norm_(loss.parameters(), 10)
            loss_optimizer.step()
            loss_optimizer.zero_grad()

    return sum(avg_loss) / (len(avg_loss) + 1e-10)


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
    parser.add_argument('--output-name', type=str, default='optimizer.pkl')
    parser.add_argument('--inner-iterations', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--feat_level', type=str, default="o,m,0.9")
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--wandb-name', type=str, default="meta-train")
    parser.add_argument('--eval-game-list', type=str, default='stable_game_list_uniform.txt')
    parser.add_argument('--cl', action="store_true")
    parser.add_argument('--learnable-loss', action='store_true', help='enable learnable loss or not')

    parser.add_argument('--use-slow-optimizer', action="store_true", help='enable slow optimizer')
    parser.add_argument('--use-slow-ema', action="store_true", help='enable slow ema')

    parser.add_argument('--slow-optimizer-freq', type=int, default=5)
    parser.add_argument('--loss-type', type=str, default='mse', choices=('mse', 'cosine'))

    args = parser.parse_args()
    main(args)