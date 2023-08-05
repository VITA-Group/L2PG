import abc
import functools
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
from torchvision import datasets, transforms
def slow_ema_update(slow_optimizer, optimizer, beta):
    for sp, p in zip(slow_optimizer.parameters(), optimizer.parameters()):
        sp.data = sp.data * beta + p.data * (1 - beta)

def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad

def main(args):

    wandb.init(project="l2o_game", name=args.wandb_name)
    wandb.config.update(args)
    torch.manual_seed(args.seed)

    cl = [50, 100, 200, 500, 1000, 2000]
    formula = args.formula.split(',')
    levels = args.feat_level.split(',')
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, args.epochs // 3)
    
    # eval_game_list = load_games_list(args.eval_game_list, args.n_player)
    best_eval_result = 1000
    best_slow_eval_result = 1000
    total_step = 0

    if args.cl:
        args.inner_iterations = cl[0]

    if args.use_slow_optimizer:
        slow_optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        slow_meta_optimizer = torch.optim.Adam(slow_optimizer.parameters(), lr=1e-3)
        slow_scheduler = torch.optim.lr_scheduler.StepLR(slow_meta_optimizer, args.epochs // 3)
    initialized = False
    for epoch in range(args.epochs):

        loss = loss_gan_high_dimension()
        gen_shapes = [(200, 200), (200, 200), (75, 200)]
        dis_shapes = [(200, 75), (200, 200), (1, 200)]
        size_gen = sum([w[0] * w[1] + w[0] for w in gen_shapes])
        size_dis = sum([w[0] * w[1] + w[0] for w in dis_shapes])
        epoch_w = [torch.randn(size_gen).cuda(), torch.randn(size_dis).cuda()]
        # gen_shapes = [(384, 64), (384, 384), (384, 384), (384, 384), (384, 384),(384, 384), (2, 384)]
        # dis_shapes = [(384, 2), (384, 384), (384, 384), (384, 384), (384, 384),(384, 384), (1, 384)]
        if args.data_cl:
            mul = epoch / args.epochs + 0.5
        else:
            mul = 1
        cur_sz = 0
        for shape in gen_shapes:
            epoch_w[0][cur_sz:cur_sz + shape[0] * shape[1]] = torch.randn(shape[0] * shape[1]) / np.sqrt(shape[0]) * (mul) * np.sqrt(1)
            cur_sz += shape[0] * shape[1]
            epoch_w[0][cur_sz:cur_sz + shape[0]] = 0
            cur_sz += shape[0]
        
        cur_sz = 0
        for shape in dis_shapes:
            epoch_w[1][cur_sz:cur_sz + shape[0] * shape[1]] = torch.randn(shape[0] * shape[1]) / np.sqrt(shape[0]) * (mul) * np.sqrt(1)
            cur_sz += shape[0] * shape[1]
            epoch_w[1][cur_sz:cur_sz + shape[0]] = 0
            cur_sz += shape[0]
        for w in epoch_w:
            w.requires_grad = True
            w.retain_grad()

        hiddens = [[torch.zeros(epoch_w[0].numel() + epoch_w[1].numel(), args.n_hidden).cuda()]]
        cells = [[torch.zeros(epoch_w[0].numel() + epoch_w[1].numel(), args.n_hidden).cuda()]]
        meta_loss = 0
        
        if (not initialized) and (epoch >= args.epochs * args.slow_optimizer_start) and (args.use_slow_optimizer):
            slow_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            initialized = True
        # print(f"init location for fast: {epoch_w}")
        transform = transforms.Compose([
            transforms.Resize(16),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))])

        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
        bs = 64
        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

        train_iter = iter(train_loader)
        if args.use_slow_optimizer and initialized:
            slow_hiddens = [[torch.zeros(epoch_w[0].numel() + epoch_w[1].numel(), args.n_hidden).cuda()]]
            slow_cells = [[torch.zeros(epoch_w[0].numel() + epoch_w[1].numel(), args.n_hidden).cuda()]]
            if args.batch_size == 1:
                slow_epoch_w = [torch.randn(size_gen).cuda(), torch.randn(size_dis).cuda()]
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
        new_grads_norm = torch.zeros(epoch_w[0].numel() + epoch_w[1].numel()).cuda()
        slow_new_grads_norm = torch.zeros(epoch_w[0].numel() + epoch_w[1].numel()).cuda()
        for iterations in range(args.inner_iterations):
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (bs, 200)))
            real_data = torch.randn(bs, 75).cuda()
            # print(real_data.shape)
            loss_partial_gen = functools.partial(loss, real_data=real_data, z=z, mode='gen')
            loss_partial_dis = functools.partial(loss, real_data=real_data, z=z, mode='dis')
            print(f'Step: {iterations}', loss_partial_gen(torch.cat([epoch_w[0], epoch_w[1]], 0)), loss_partial_dis(torch.cat([epoch_w[0], epoch_w[1]], 0)))
            weights = torch.cat([epoch_w[0], epoch_w[1]], 0)
            # print(loss_partial_dis(torch.cat([epoch_w[0], epoch_w[1]], 0)))

            grad_L = [[torch.autograd.grad(loss_partial_gen(weights), epoch_w[0], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(weights), epoch_w[0], create_graph=True)[0]], [torch.autograd.grad(loss_partial_gen(weights), epoch_w[1], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(weights), epoch_w[1], create_graph=True)[0]]]
            grads = torch.cat([grad_L[0][0],grad_L[1][1]])
            ham = torch.dot(grads, grads.detach())
            
            H_t_xi = torch.cat([get_gradient(ham, epoch_w[i]) for i in range(2)]).detach()
            H_xi = torch.cat([get_gradient(sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
                for j in range(2)]), epoch_w[i]) for i in range(2)]).detach()
        
            Sg = (H_xi + H_t_xi) / 2
            Ag = (H_t_xi - H_xi) / 2
            obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)]                     
            obs = torch.cat(obs, 1).detach()
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
                wandb.log({"step_loss_reg_1": loss_reg_1}, step=total_step)

            elif args.reg_2:
                
                H = torch.sum(grads ** 2) / 2
                dh = []
                for g, w in zip(grads, epoch_w):
                    dh.append(torch.autograd.grad(H, w, retain_graph=True)[0])
                dh = torch.cat(dh).view(-1, 1)
                reg_2s = torch.sign((grads.view(1,-1) @ dh.view(-1,1)).sum())
                    
            update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
            # print(iterations, epoch_w, update)

            new_grad = (update[:,0] * grads[:] - update[:,1] * Ag[:] - update[:,2] * Sg[:])
            new_grads_norm = new_grads_norm * 0.9 + (new_grad.detach() ** 2) * 0.1
            normalized = new_grad / torch.sqrt(new_grads_norm + 1e-8)
            if args.batch_size == 1:
                epoch_w[0] = epoch_w[0] - normalized[:epoch_w[0].shape[0]] * 1e-4
                epoch_w[1] = epoch_w[1] - normalized[epoch_w[0].shape[0]:] * 1e-4
            else:
                for idx, epoch_w in enumerate(epoch_ws):
                    for j in range(args.n_player):
                        para_idx = j + args.n_player * idx
                        epoch_w[j] = epoch_w[j] - (update[para_idx, 0] * grads[para_idx] - update[para_idx, 1] * Ag[para_idx] - update[para_idx, 2] * Sg[para_idx]) * scale[0]
                    # epoch_w[1] = epoch_w[1] - (update[1 + args.n_player * idx, 0] * grads[1] - update[1,1] * Ag[1] - update[1,2] * Sg[1]) * scale[1]
            new_hs.append(new_h)
            new_cs.append(new_c)
            # print(f"updated location for fast: {epoch_w}")
            hiddens = new_hs
            cells = new_cs
            if args.normalize_meta_loss:
                step_meta_loss = torch.sum(grads ** 2) / torch.sum(init_grads ** 2) 
            else:
                if not args.reg_2:
                    step_meta_loss = 1 / 2 * torch.sum(grads ** 2) / args.batch_size
                else:
                    step_meta_loss = torch.sum((grads ** 2).view(args.batch_size, -1) * reg_2s.view(args.batch_size, -1), 1)
                    step_meta_loss = step_meta_loss[step_meta_loss > -100]
                    step_meta_loss = torch.mean(step_meta_loss) / 2
            total_step = total_step + 1
            wandb.log({"step_meta_loss": step_meta_loss}, step=total_step)
            meta_loss += step_meta_loss
            if args.reg_1:
                if loss_reg_1 < 0:
                    meta_loss -= loss_reg_1 * args.reg_coef
            if (meta_loss > 10000) or (meta_loss < -1e4):
                break
            elif (iterations + 1) % args.unroll_length == 0:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer.parameters(), 1)
                meta_optimizer.step()
                optimizer.zero_grad()
                print(iterations + 1, f'meta loss: {meta_loss.item()}', f'current loss: {torch.sum(grads ** 2)}')
                wandb.log({"meta_loss": meta_loss}, step=total_step)
                print(f'Step: {iterations}', loss_partial_gen(weights), loss_partial_dis(weights))
                meta_loss = 0
                hiddens = tree.map_structure(detach, hiddens)
                cells = tree.map_structure(detach, cells)
                epoch_w = tree.map_structure(detach, epoch_w)
                del grad_L
                del H_t_xi
                del H_xi

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
                        slow_epoch_w = tree.map_structure(detach, slow_epoch_w)
                    loss_partial_gen = functools.partial(loss, real_data=real_data, z=z, mode='gen')
                    loss_partial_dis = functools.partial(loss, real_data=real_data, z=z, mode='dis')
                    slow_weights = torch.cat([slow_epoch_w[0], slow_epoch_w[1]], 0)
                    slow_grad_L = [[torch.autograd.grad(loss_partial_gen(slow_weights), slow_epoch_w[0], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(slow_weights), slow_epoch_w[0], create_graph=True)[0]], [torch.autograd.grad(loss_partial_gen(slow_weights), slow_epoch_w[1], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(slow_weights), slow_epoch_w[1], create_graph=True)[0]]]
                    slow_grads = torch.cat([slow_grad_L[0][0],slow_grad_L[1][1]])
                    slow_ham = torch.dot(slow_grads, slow_grads.detach())
                    
                    slow_H_t_xi = torch.cat([get_gradient(slow_ham, slow_epoch_w[i]) for i in range(2)]).detach()
                    slow_H_xi = torch.cat([get_gradient(sum([torch.dot(slow_grad_L[j][i], slow_grad_L[j][j].detach())
                        for j in range(2)]), slow_epoch_w[i]) for i in range(2)]).detach()
                
                    slow_Sg = (slow_H_xi + slow_H_t_xi) / 2
                    slow_Ag = (slow_H_t_xi - slow_H_xi) / 2
                    slow_obs = [slow_grads.view(-1, 1), slow_Ag.view(-1, 1), slow_Sg.view(-1, 1)]                     
                    slow_obs = torch.cat(slow_obs, 1).detach()
                    slow_stats = init_stats(slow_obs, feat_levels=levels)
                    slow_obs, slow_stats = construct_obs(slow_obs, levels, slow_stats, iterations // args.slow_optimizer_freq)  
                    
                    slow_new_hs = []
                    slow_new_cs = []
                    
                    slow_update, slow_scale, slow_new_h, slow_new_c = slow_optimizer(slow_obs, slow_hiddens[0], slow_cells[0])
                    slow_new_grad = (slow_update[:,0] * slow_grads[:] - slow_update[:,1] * slow_Ag[:] - slow_update[:,2] * slow_Sg[:])
                    slow_new_grads_norm = slow_new_grads_norm * 0.9 + (slow_new_grad.detach() ** 2) * 0.1
                    slow_normalized = slow_new_grad / torch.sqrt(slow_new_grads_norm + 1e-8)
                    if args.batch_size == 1:
                        slow_epoch_w[0] = slow_epoch_w[0] - slow_normalized[:slow_epoch_w[0].shape[0]] * 2e-4
                        slow_epoch_w[1] = slow_epoch_w[1] - slow_normalized[slow_epoch_w[0].shape[0]:] * 2e-4
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
                    
  
        if (epoch + 1) % 50 == 0:    
            torch.save({"state_dict": optimizer.state_dict()}, args.output_name + f"_{epoch}")
            if args.use_slow_optimizer:
                torch.save({"state_dict": slow_optimizer.state_dict()}, args.output_name + f"_{epoch}_slow")
            if args.cl:
                try:
                    args.inner_iterations = cl[cl.index(args.inner_iterations) + 1]
                except IndexError:
                    pass

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
        loss = loss_gan()
        ws = []
        initial_w = [torch.randn(64 * 64 * 5 + 128).cuda() * 0.1, torch.randn(64 * 64 * 4 + 3 * 64).cuda() * 0.1]
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

    args = parser.parse_args()
    main(args)