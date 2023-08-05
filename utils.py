
import torch
import numpy as np

def generate_game_sample(args, epoch=0):
    if args.game_distribution == 'gaussian':
        rng = np.random.randn
    elif args.game_distribution == 'negative-uniform': 
        rng = lambda x: 2 * (np.random.randn(x)) - 1
    else:
        rng = np.random.rand

    if args.stable:
        a = rng(args.n_player ** 2 + args.n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        while w[0] < 0 or w[1] < 0:
            a = rng(args.n_player ** 2 + args.n_player)
            w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
    elif args.stable_saddle:
        a = rng(args.n_player ** 2 + args.n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        while w[0] < 0:
            a = rng(args.n_player ** 2 + args.n_player)
            w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
    elif args.data_cl:
        print("CL mode")
        a = rng(args.n_player ** 2 + args.n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        print(f"Current Threshold: {(1 - epoch / args.epochs * 0.1)}")
        is_unstable = np.random.rand() > (1 - epoch / args.epochs * 0.1)

        if is_unstable:
            print("Unstable")
            while (w[0] > 0 or w[1] > 0):
                a = rng(args.n_player ** 2 + args.n_player)
                w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        else:
            while (w[0] < 0 or w[1] < 0):
                a = rng(args.n_player ** 2 + args.n_player)
                w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
    else:
        a = rng(args.n_player ** 2 + args.n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        is_unstable = np.random.rand() > 0.2

        if is_unstable:
            while (w[0] > 0 or w[1] > 0):
                a = rng(args.n_player ** 2 + args.n_player)
                w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
        else:
            while (w[0] < 0 or w[1] < 0):
                a = rng(args.n_player ** 2 + args.n_player)
                w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(args.n_player, args.n_player))
                
    return a

    
def load_games_list(game_list, n_player):
    lines = open(game_list).readlines()
    array_lines = []
    for line in lines:
        line_array = np.array(list(map(float, line.strip().split(','))))
        array_lines.append(line_array)
        w, v = np.linalg.eig(np.array([line_array[0], (line_array[2] + line_array[3]) / 2, (line_array[2] + line_array[3]) / 2, line_array[1]]).reshape(n_player, n_player))
        print(w)
    return array_lines


def construct_obs(obs, feat_levels, stats, step):
    final_obs = []
    if 'o' in feat_levels:
        final_obs.append(obs)
    if 'm0.5' in feat_levels:
        stats['m0.5'] = stats['m0.5'] * 0.5 + obs * 0.5
        final_obs.append(stats['m0.5'])
    if 'm0.9' in feat_levels:
        stats['m0.9'] = stats['m0.9'] * 0.9 + obs * 0.1
        final_obs.append(stats['m0.9'])
    if 'm0.99' in feat_levels:
        stats['m0.99'] = stats['m0.99'] * 0.99 + obs * 0.01
        final_obs.append(stats['m0.99'])
    
    stats['mt'] = 0.95 * stats['mt'] + 0.05 * obs
    stats['vt'] = 0.9 * stats['vt'] + 0.1 * (obs ** 2)
    mt_hat = stats['mt'] / (1 - (0.95 ** (step + 1)))
    vt_hat = stats['vt'] / (1 - (0.95 ** (step + 1)))
    vs = torch.sqrt(vt_hat) + 1e-8
    mt_tilde = mt_hat / vs
    gt_tilde = obs / vs

    if 'mt' in feat_levels:
        final_obs.append(mt_tilde)
    if 'gt' in feat_levels:
        final_obs.append(gt_tilde)
    if 't' in feat_levels:
        final_obs.append(torch.tensor([step / 10.]).cuda().view(-1, 1).repeat(obs.shape[0], 1))
    # print(final_obs)
    return torch.cat(final_obs, 1), stats


def init_stats(obs, feat_levels):
    stats = {}
    for key in ['m0.5', 'm0.9', 'm0.99']:
        if key in feat_levels:
            stats[key] = torch.zeros_like(obs)
    stats['mt'] = torch.zeros_like(obs)
    stats['vt'] = torch.zeros_like(obs)
    return stats

def detach(x):
    x.detach_()
    x.requires_grad = True
    return x


def random_unit(n):
    x = torch.cos(torch.FloatTensor([n])).cuda()
    y = torch.sin(torch.FloatTensor([n])).cuda()
    return [x, y]

def random_unit_four(n1, n2, n3):
    x = torch.cos(torch.FloatTensor([n1])).cuda()
    y = torch.sin(torch.FloatTensor([n1])).cuda() * torch.cos(torch.FloatTensor([n2])).cuda()
    w = torch.sin(torch.FloatTensor([n1])).cuda() * torch.sin(torch.FloatTensor([n2])).cuda()  * torch.sin(torch.FloatTensor([n3])).cuda()
    z = torch.sin(torch.FloatTensor([n1])).cuda() * torch.sin(torch.FloatTensor([n2])).cuda()  * torch.cos(torch.FloatTensor([n3])).cuda()

    return [x, y, w, z]

def random_ball_four(n1, n2, n3):
    x = torch.cos(torch.FloatTensor([n1])).cuda()
    y = torch.sin(torch.FloatTensor([n1])).cuda() * torch.cos(torch.FloatTensor([n2])).cuda()
    w = torch.sin(torch.FloatTensor([n1])).cuda() * torch.sin(torch.FloatTensor([n2])).cuda()  * torch.sin(torch.FloatTensor([n3])).cuda()
    z = torch.sin(torch.FloatTensor([n1])).cuda() * torch.sin(torch.FloatTensor([n2])).cuda()  * torch.cos(torch.FloatTensor([n3])).cuda()
    scale = np.random.rand()
    return [x * scale, y * scale, w * scale, z * scale]

def random_ball(n):
    length = np.random.rand()
    x = torch.cos(torch.FloatTensor([n])).cuda() * length
    y = torch.sin(torch.FloatTensor([n])).cuda() * length
    return [x, y]


def init_weight(mode='unit'):
    if mode == 'unit':
        players_w = random_unit(np.random.rand() * 1000)
        initial_w = list(players_w)
    elif mode == 'ball':
        players_w = random_ball(np.random.rand() * 1000)
        initial_w = list(players_w)
    return initial_w


def init_weight_four(mode='unit'):
    if mode == 'unit':
        players_w = random_unit_four(np.random.rand() * 1000, np.random.rand() * 1000, np.random.rand() * 1000)
        initial_w = list(players_w)
    elif mode == 'ball':
        players_w = random_ball_four(np.random.rand() * 1000, np.random.rand() * 1000, np.random.rand() * 1000)
        initial_w = list(players_w)
    return initial_w


