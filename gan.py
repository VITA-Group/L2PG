import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import matplotlib.pyplot as plt
from meta_module import *
import argparse

import functools
# from functorch import make_functional, hvp, 
def rgetattr(obj, attr, *args):
	def _getattr(obj, attr):
			return getattr(obj, attr, *args)
	return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
	pre, _, post = attr.rpartition('.')
	return setattr(rgetattr(obj, pre) if pre else obj, post, val)

class MLP(MetaModule):
  """An MLP with hidden layers of the same width as the input."""

  def __init__(self, depth, hidden_size, out_dim, input_dim):
    super(MLP, self).__init__()
    self._depth = depth
    self._hidden_size = hidden_size
    self._out_dim = out_dim
    self.model = MetaSequential(
        MetaLinear(input_dim, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        MetaLinear(self._hidden_size, self._out_dim)
    )
  def forward(self, x):
    return self.model(x)

def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    plt.show()

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

    generator = MLP(6, 384, 2, 64).cuda()
    discriminator = MLP(6, 384, 1, 2).cuda()

    formula = args.formula.split(',')
    levels = args.feat_level.split(',')
    optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, args.epochs // 3)
    
    # eval_game_list = load_games_list(args.eval_game_list, args.n_player)
    best_eval_result = 1000
    total_step = 0

    if args.cl:
        args.inner_iterations = cl[0]

    if args.use_slow_optimizer:
        slow_optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        slow_meta_optimizer = torch.optim.Adam(slow_optimizer.parameters(), lr=1e-3)
        slow_scheduler = torch.optim.lr_scheduler.StepLR(slow_meta_optimizer, args.epochs // 3)
    initialized = False
    for epoch in range(args.epochs):
        generator = MLP(6, 384, 2, 64)
        discriminator = MLP(6, 384, 1, 2)
        
        flatten_g_weight = []
        flatten_d_weight = []
        g_size = 0
        d_size = 0

        for weight in generator.parameters():
            flatten_g_weight.append(weight.view(-1))
            g_size += weight.numel()
        for weight in discriminator.parameters():
            flatten_d_weight.append(weight.view(-1))
            d_size += weight.numel()

        hiddens = [[torch.zeros(g_size + d_size, args.n_hidden).cuda()]]
        cells = [[torch.zeros(g_size + d_size, args.n_hidden).cuda()]]
        meta_loss = 0

        if (not initialized) and (epoch >= args.epochs * args.slow_optimizer_start) and (args.use_slow_optimizer):
            slow_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            initialized = True
        # print(f"init location for fast: {epoch_w}")
        if args.use_slow_optimizer and initialized:
            slow_hiddens = [[torch.zeros(g_size + d_size, args.n_hidden).cuda()]]
            slow_cells = [[torch.zeros(g_size + d_size, args.n_hidden).cuda()]]
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

        for iterations in range(args.inner_iterations):
            
            
            real_data = x_real_builder(256).float()
            # print(real_data.shape)
            fake_data = generator(torch.randn(256, 64).cuda())
            disc_out_real = discriminator(real_data)
            disc_out_fake = discriminator(fake_data)
            disc_loss_real = torch.nn.BCEWithLogitsLoss()(disc_out_real, torch.ones(256, 1).cuda())
            disc_loss_fake = torch.nn.BCEWithLogitsLoss()(disc_out_fake, torch.zeros(256, 1).cuda())
            gen_loss = torch.nn.BCEWithLogitsLoss()(disc_out_fake, torch.ones(256, 1).cuda())
            g_weight = []
            d_weight = []
            for weight in generator.parameters():
                g_weight.append(weight)
                # g_size += weight.numel()
            for weight in discriminator.parameters():
                d_weight.append(weight)
                # d_size += weight.numel()
            disc_grads = torch.autograd.grad(disc_loss_real + disc_loss_fake, d_weight, create_graph=True, retain_graph=True)
            gen_grads = torch.autograd.grad(gen_loss, g_weight, create_graph=True, retain_graph=True)
            # disc_grads_flat = tree.map_structure(lambda x: x.flatten(), disc_grads)
            # gen_grads_flat = tree.map_structure(lambda x: x.flatten(), gen_grads)

            # grads = torch.cat([torch.cat(disc_grads_flat, 0), torch.cat(gen_grads_flat, 0)], 0)
            print(torch.autograd.grad(disc_grads[-1], d_weight[0]))
            print(torch.autograd.functional.jacobian(lambda x: disc_grads[-1], d_weight[-1]))
            # a, b = torch.autograd.functional.jvp(lambda *x: grads, tuple(d_weight + g_weight), tuple(disc_grads + gen_grads))
            # print(b.mean())
            assert False
            Jacob = [[0 for _ in range(g_size + d_size)] for _ in range(g_size + d_size)]
            for i in range(g_size + d_size):
                cur_size = 0
                Jacob[i][:d_size] = torch.cat(torch.autograd.grad(grads[i], d_weight, retain_graph=True), 0)
                Jacob[i][d_size:] = torch.cat(torch.autograd.grad(grads[i], g_weight, retain_graph=True), 0)
            jacob = torch.tensor(Jacob).cuda()
            S = (jacob + torch.transpose(jacob, 0, 1)) / 2
            print(S)
            A = (jacob - torch.transpose(jacob, 0, 1)) / 2
            print(A)
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
            '''
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
                dh = torch.stack(dh)
                reg_2 = torch.sign((grads.view(1,-1) @ dh.view(-1,1)).sum())
                wandb.log({"step_reg_2": loss_reg_1}, step=total_step)
            '''    
            update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
            # print(iterations, epoch_w, update)
            result_params_dis = {}
            idx = 0
            for name, p in discriminator.named_parameters():
                size = p.nelement()
                result_params_dis[name] = p - update[idx:idx+size, 0] * grads[idx:idx+size].view(*p.size()) - update[idx:idx+size, 1] * Ag[idx:idx+size].view(*p.size()) - update[idx:idx+size, 2] * Sg[idx:idx+size].view(*p.size())
                result_params_dis[name].retain_grad()        
                idx = idx + size
            result_params_gen = {}
            for name, p in generator.named_parameters():
                size = p.nelement()
                result_params_gen[name] = p - update[idx:idx+size, 0] * grads[idx:idx+size].view(*p.size()) - update[idx:idx+size, 1] * Ag[idx:idx+size].view(*p.size()) - update[idx:idx+size, 2] * Sg[idx:idx+size].view(*p.size())
                result_params_gen[name].retain_grad()        
                idx = idx + size
            
            new_hs.append(new_h)
            new_cs.append(new_c)
            hiddens = new_hs
            cells = new_cs
            reg_2 = 1
            if not args.reg_2:
                step_meta_loss = 1 / 2 * torch.sum(grads ** 2) 
            else:
                step_meta_loss = torch.mean((grads ** 2).view(args.batch_size, -1) * reg_2) / 2
            total_step = total_step + 1
            wandb.log({"step_meta_loss": step_meta_loss}, step=total_step)
            meta_loss += step_meta_loss
            if args.reg_1:
                if loss_reg_1 < 0:
                    meta_loss -= loss_reg_1 * args.reg_coef
                        
            if (meta_loss > 1e5) or (meta_loss < -1e5):
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
                for name in result_params_dis:
                    param = result_params_dis[name].detach()
                    param.requires_grad_()
                    rsetattr(discriminator, name, param)
                for name in result_params_gen:
                    param = result_params_gen[name].detach()
                    param.requires_grad_()
                    rsetattr(generator, name, param)
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
                slow_eval_result = evaluate(slow_optimizer, eval_game_list, formula, levels, args, slow=True)
                torch.save({"state_dict": slow_optimizer.state_dict(), "eval_results": mean_eval_result}, "slow_" + args.output_name)
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
        hiddens = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
        cells = [[torch.zeros(w.numel() * args.n_player * args.batch_size, args.n_hidden).cuda()]]
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


def x_real_builder(batch_size):
    sigma = 0.1
    skel = np.array([
        [ 1.50,  1.50],
        [ 1.50,  0.50],
        [ 1.50, -0.50],
        [ 1.50, -1.50],
        [ 0.50,  1.50],
        [ 0.50,  0.50],
        [ 0.50, -0.50],
        [ 0.50, -1.50],
        [-1.50,  1.50],
        [-1.50,  0.50],
        [-1.50, -0.50],
        [-1.50, -1.50],
        [-0.50,  1.50],
        [-0.50,  0.50],
        [-0.50, -0.50],
        [-0.50, -1.50],
    ])
    temp = torch.from_numpy(np.tile(skel, (batch_size // 16 + 1,1)))
    mus = temp[0:batch_size,:]
    return (mus + sigma * torch.randn(batch_size, 2) * 0.2).cuda()


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