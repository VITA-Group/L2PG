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
from torchvision.utils import save_image

def slow_ema_update(slow_optimizer, optimizer, beta):
    for sp, p in zip(slow_optimizer.parameters(), optimizer.parameters()):
        sp.data = sp.data * beta + p.data * (1 - beta)

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
bs = 256
temp = np.tile(skel, (bs // 16 + 1,1))
mus = temp[0:bs,:]
    
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
    
    checkpoints = torch.load(args.output_name)
    optimizer.load_state_dict(checkpoints['state_dict'])
    evaluate(optimizer, None, formula, levels, args)

import matplotlib.pyplot as plt
from scipy import stats
def kde(mu, tau, step, bbox=None, xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = stats.gaussian_kde(values)
    print(kernel.d)
    print(kernel.n)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    f = np.reshape(kernel(positions).T, xx.shape)

    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    plt.savefig(f'sga_gen_{step}.png')

def evaluate(net, game_list, formula, levels, args, slow=False):
    counts = []
    if True:
        optimizer = RNNOptimizer(True, args.n_hidden, 10, False, n_features=len(formula) * (len(levels)), no_tanh=args.no_tanh).cuda()
        optimizer.load_state_dict(net.state_dict())
        lrs = []
        ws = []
        updates = []
        grads_ = []
        Ags = []
        Sgs = []
        loss = loss_gan_mnist()
        ws = []
        gen_shapes = [(32, 64), (64, 32), (128, 64), (256, 128)]
        dis_shapes = [(128, 256), (64, 128), (32, 64), (1, 32)]
        size_gen = sum([w[0] * w[1] + w[0] for w in gen_shapes])
        size_dis = sum([w[0] * w[1] + w[0] for w in dis_shapes])
        epoch_w = [torch.randn(size_gen).cuda(), torch.randn(size_dis).cuda()]
        cur_sz = 0
        for shape in gen_shapes:
            epoch_w[0][cur_sz:cur_sz + shape[0] * shape[1]] = torch.randn(shape[0] * shape[1]) / np.sqrt(shape[0])
            cur_sz += shape[0] * shape[1]
            epoch_w[0][cur_sz:cur_sz + shape[0]] = 0
            cur_sz += shape[0]
        
        cur_sz = 0
        for shape in dis_shapes:
            epoch_w[1][cur_sz:cur_sz + shape[0] * shape[1]] = torch.randn(shape[0] * shape[1]) / np.sqrt(shape[0])
            cur_sz += shape[0] * shape[1]
            epoch_w[1][cur_sz:cur_sz + shape[0]] = 0
            cur_sz += shape[0]
        acount = 0
        initial_w = [epoch_w]
        bs = 10
        z_eval = torch.cuda.FloatTensor(np.random.normal(0, 1, (bs * 10, 64)))
        for wi in initial_w:
            players_w = wi
            ws.append(list(wi))
            losses = []
            for w in players_w:
                w.requires_grad = True
                w.retain_grad()
                w.cuda()

            hiddens = [[torch.zeros(players_w[0].numel() + players_w[1].numel(), args.n_hidden).cuda()]]
            cells = [[torch.zeros(players_w[0].numel() + players_w[1].numel(), args.n_hidden).cuda()]]
            count = 0
            def get_gradient(function, param):
                grad = torch.autograd.grad(function, param, create_graph=True)[0]
                return grad
            new_grads_norm = torch.zeros(epoch_w[0].numel() + epoch_w[1].numel()).cuda()
            new_grads_momentum = torch.zeros(epoch_w[0].numel() + epoch_w[1].numel()).cuda()
            
            transform = transforms.Compose([
                transforms.Resize(16),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

            train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
            bs = 100
            # Data Loader (Input Pipeline)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
            while count < 100000:
                z = torch.cuda.FloatTensor(np.random.normal(0, 1, (bs, 64)))
                try:
                    real_data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    real_data = next(train_iter)

                real_data, _ = real_data
                real_data = real_data.view(-1, 256).cuda()

                loss_partial_gen = functools.partial(loss, real_data=real_data, z=z, mode='gen')
                loss_partial_dis = functools.partial(loss, real_data=real_data, z=z, mode='dis')
                
                weights = torch.cat([players_w[0], players_w[1]], 0)
                print(f'Step: {count}', loss_partial_gen(weights), loss_partial_dis(weights))

                grad_L = [[torch.autograd.grad(loss_partial_gen(weights), players_w[0], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(weights), players_w[0], create_graph=True)[0]], [torch.autograd.grad(loss_partial_gen(weights), players_w[1], create_graph=True)[0], torch.autograd.grad(loss_partial_dis(weights), players_w[1], create_graph=True)[0]]]
                grads = torch.cat([grad_L[0][0],grad_L[1][1]])
                ham = torch.dot(grads, grads.detach())
                
                H_t_xi = torch.cat([get_gradient(ham, players_w[i]) for i in range(2)]).detach()
                H_xi = torch.cat([get_gradient(sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
                    for j in range(2)]), players_w[i]) for i in range(2)]).detach()
                Sg = (H_xi + H_t_xi) / 2
                Ag = (H_t_xi - H_xi) / 2
                Sg = Sg.detach()
                Ag = Ag.detach()
                # grads = grads.detach() + Ag.detach()
                obs = [grads.view(-1, 1), Ag.view(-1, 1), Sg.view(-1, 1)] 
                obs = torch.cat(obs, 1).detach()
                if count == 0:
                    stats = init_stats(obs, feat_levels=levels)
                obs, stats = construct_obs(obs, levels, stats, count)
                new_hs = []
                new_cs = []                    
                with torch.no_grad():
                    update, scale, new_h, new_c = optimizer(obs, hiddens[0], cells[0])
                    # pass
                # g = grads / torch.sqrt(stats['vt'][:, 0] + 1e-8)
                if count <= 2000:
                    new_grad = (update[:,0] * grads[:].detach() - update[:,1] * Ag[:].detach() - update[:,2] * Sg[:].detach() )
                    new_grads_norm = new_grads_norm * 0.9 + (new_grad.detach() ** 2) * 0.1
                    normalized = new_grad / torch.sqrt(new_grads_norm + 1e-8).detach()
                    players_w[0] = players_w[0] - normalized[:players_w[0].shape[0]] * 1.5e-4
                    players_w[1] = players_w[1] - normalized[players_w[0].shape[0]:] * 1.5e-4
                else:
                    new_grad = (grads.detach() + Ag.detach())
                    new_grads_norm = new_grads_norm * 0.9 + (new_grad.detach() ** 2) * 0.1
                    normalized = new_grad / torch.sqrt(new_grads_norm + 1e-8).detach()
                    players_w[0] = players_w[0] - normalized[:players_w[0].shape[0]] * 2e-4
                    players_w[1] = players_w[1] - normalized[players_w[0].shape[0]:] * 2e-4

                new_hs.append(new_h)
                new_cs.append(new_c)
                hiddens = new_hs
                cells = new_cs
                count += 1
                hiddens = tree.map_structure(detach, hiddens)
                cells = tree.map_structure(detach, cells)
                players_w = tree.map_structure(detach, players_w)
                del grad_L
                del grads
                del H_t_xi
                del H_xi
                del ham
                if (count + 1) % 500 == 0:

                    cur_sz = 0
                    weights_gen_split = []
                    for shape in gen_shapes:
                        weights_gen_split.append(players_w[0][cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
                        cur_sz += shape[0] * shape[1]
                        weights_gen_split.append(players_w[0][cur_sz:cur_sz + shape[0]].view(shape[0]))
                        cur_sz += shape[0]
   
                    with torch.no_grad():
                        z = F.linear(z_eval, weights_gen_split[0], weights_gen_split[1])

                        for i in range(2, len(weights_gen_split), 2):
                            z = F.relu(z)
                            z = F.linear(z, weights_gen_split[i], weights_gen_split[i + 1])
                    # torch.save(z, {})
                    z = F.tanh(z)
                    z = z.detach().cpu().view(z.size(0), 1, 16, 16) 
                    save_image(z, f"{count + 1}_l2o.png", nrow=10)
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