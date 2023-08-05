import torch
import torch.nn as nn

def loss_9(n_action, epsilon=0.1):
    # n_action for type check
    assert n_action == 1
    def l(players_w):
        l1 = -epsilon / 2 * (players_w[0] ** 2) - players_w[0] * players_w[1]
        l2 = -epsilon / 2 * (players_w[1] ** 2) + players_w[0] * players_w[1]
        return [l1, l2]
    return l

def loss_sym(n_action, epsilon=0.1):
    # n_action for type check
    def l(players_w):
        l1 = (players_w[0] * players_w[1]).sum()
        l2 = - (players_w[0] * players_w[1]).sum()
        return [l1, l2]
    return l

def loss_balduzzi(n_action, epsilon=0.1):
    # n_action for type check
    assert n_action == 1
    def l(players_w):
        l1 = 1 / 2 * (players_w[0] ** 2) + 10 * players_w[0] * players_w[1]
        l2 = 1 / 2 * (players_w[1] ** 2) - 10 * players_w[0] * players_w[1]
        return [l1, l2]
    return l

import math
def loss_seesaw(n_action, epsilon=0.1):
    # n_action for type check
    assert n_action == 1
    def l(players_w):
        l1 = -0.95 * players_w[1] * torch.sin(0.95 * math.pi * players_w[0])
        l2 = 0.95 * players_w[1] * torch.sin(0.95 * math.pi * players_w[0])
        return [l1, l2]
    return l

def grad(loss, players_w):
    loss_values = loss(players_w)
    n_player = len(players_w)
    grads = []
    for i in range(n_player):
        grads.append(torch.autograd.grad(loss_values[i], players_w[i], retain_graph=True, create_graph=True)[0])
    return torch.cat(grads, 0)

def batch_grad(losses, players_w):
    all_grads = []
    for loss, w in zip(losses, players_w):
        loss_values = loss(players_w)
        n_player = len(players_w)
        grads = []
        for i in range(n_player):
            grads.append(torch.autograd.grad(loss_values[i], players_w[i], retain_graph=True, create_graph=True)[0])
        all_grads.append(torch.cat(grads))
    return all_grads

def decompose(grads, players_w):
    n_player = len(players_w)
    Jacob = [[0 for _ in range(n_player)] for _ in range(n_player)]
    for i in range(n_player):
        for j in range(n_player):
            Jacob[i][j] = torch.autograd.grad(grads[i], players_w[j], retain_graph=True)[0]

    jacob = torch.tensor(Jacob).cuda()
    S = (jacob + torch.transpose(jacob, 0, 1)) / 2
    A = (jacob - torch.transpose(jacob, 0, 1)) / 2
    return S, A

def setup_losses(args):
    # TODO: add loss type in args
    return []


def loss_quadratic(n_action, S1=None, S2=None, M12=None, M21=None, b1=None, b2=None):

    if S1 is None:
        S1 = torch.randn((n_action, n_action)).cuda()
    if S2 is None:
        S2 = torch.randn((n_action, n_action)).cuda()
    if M12 is None:
        M12 = torch.randn((n_action, n_action)).cuda()
    if M21 is None:
        M21 = torch.randn((n_action, n_action)).cuda()
    if b1 is None:
        b1 = torch.randn(n_action).cuda()
    if b2 is None:
        b2 = torch.randn(n_action).cuda()
        
    def l(players_w):
        l1 = 1 / 2 * (players_w[0] * S1 * players_w[0].T) + players_w[0] * M12 * players_w[1] + players_w[0] * b1
        l2 = 1 / 2 * (players_w[1] * S2 * players_w[1].T) + players_w[0] * M21 * players_w[1] + players_w[1] * b2
        return [l1, l2]
    return l, {'S1': S1, 'S2': S2, 'M12': M12, 'M21': M21, 'b1': b1, 'b2': b2}

def loss_ours(n_action, a=None, b=None):

    if a is None:
        a = torch.rand(1).cuda() * 0.9 + 0.1
    if b is None:
        b = torch.rand(1).cuda() * 0.9 + 0.1
        
        
    def l(players_w):
        l1 = 1 / 3 * a * (players_w[0] ** 3) - 1 / 2 * b * (players_w[0] ** 2)
        l2 = 1 / 3 * a * (players_w[1] ** 3) - 1 / 2 * b * (players_w[1] ** 2)
        return [l1, l2]
    return l, None

def loss_four(n_action, epi):
    def l(players_w):
        l1 = epi / 2 * players_w[0] ** 2 + players_w[0] * players_w[1] + players_w[0] * players_w[2] + players_w[0] * players_w[3]
        l2 = epi / 2 * players_w[1] ** 2 - players_w[0] * players_w[1] + players_w[1] * players_w[2] + players_w[1] * players_w[3]
        l3 = epi / 2 * players_w[2] ** 2 - players_w[0] * players_w[2] - players_w[1] * players_w[2] + players_w[2] * players_w[3]
        l4 = epi / 2 * players_w[3] ** 2 - players_w[0] * players_w[3] - players_w[1] * players_w[3] - players_w[2] * players_w[3]
        return [l1, l2, l3, l4]
    return l, None

def forward_dis(weight, _input):
    count = 0
    W_sizes = (2*384, 384*384, 384*384, 384*384, 384*384, 384*384, 384*1)
    out_channels = (384, 384, 384, 384, 384, 384, 1)
    W = []
    x = _input
    for i in range(len(W_sizes)):
        w = weight[count:count+W_sizes[i]].reshape(-1, out_channels[i])
        x = x @ w
    
    return x
    

def forward_gen(weight, _input):
    count = 0
    W_sizes = (64*384, 384*384, 384*384, 384*384, 384*384, 384*384, 384*2)
    out_channels = (384, 384, 384, 384, 384, 384, 2)
    W = []
    x = _input
    for i in range(len(W_sizes)):
        w = weight[count:count+W_sizes[i]].reshape(-1, out_channels[i])
        x = x @ w
    
    return x

import numpy as np    

def loss_det(w, x_fake):
    x_real = x_real_builder(64).float()
    out_real = forward_dis(w, x_real)
    out_fake = forward_dis(w, x_fake)
    out = torch.cat([out_real, out_fake], 0).view(-1)
    labels = torch.FloatTensor([1] * out_real.shape[0] + [0] * out_fake.shape[0])
    return torch.nn.functional.binary_cross_entropy_with_logits(out, labels, reduction='mean') * 2

def loss_gen(w_gen, w_det):
    pass
if __name__ == "__main__":
    loss = loss_9(1, 0.1)
    players_w = [torch.randn(1) for _ in range(2)]
    for w in players_w:
        w.requires_grad = True
        w.retain_grad()
    grads = grad(loss, players_w)
    print(players_w)
    print(grads)

    assert (-0.1 * players_w[0] - players_w[1]) == grads[0]
    assert (-0.1 * players_w[1] + players_w[0]) == grads[1]

    S, A = decompose(grads, players_w)

    # print(x_real_builder(16).shape)

    w1 = torch.zeros(738432)
    w2 = torch.zeros(762624)
    # print(forward_dis(w1, torch.zeros(20,2)))
    # print(forward_gen(w2, torch.zeros(20,64)))

    fake = torch.randn(64,2)
    print(loss_det(w1, fake))

import torch.nn.functional as F

def loss_gan():

    def l(weights, real_data, z, mode):
        gen_shapes = [(384, 64), (384, 384), (384, 384), (384, 384), (384, 384), (384, 384), (2, 384)]
        dis_shapes = [(384, 2), (384, 384), (384, 384), (384, 384), (384, 384), (384, 384), (1, 384)]

        weights_gen_split = []
        cur_sz = 0
        for shape in gen_shapes:
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]

        weights_dis_split = []
        for shape in dis_shapes:
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]    
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]
        
        z = F.linear(z, weights_gen_split[0], weights_gen_split[1])

        for i in range(2, len(weights_gen_split), 2):
            z = F.relu(z)
            z = F.linear(z, weights_gen_split[i], weights_gen_split[i + 1])

        fake_out = F.linear(z, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            fake_out = F.relu(fake_out)
            fake_out = F.linear(fake_out, weights_dis_split[i], weights_dis_split[i + 1])
        
        
        real_out = F.linear(real_data, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            real_out = F.relu(real_out)
            real_out = F.linear(real_out, weights_dis_split[i], weights_dis_split[i + 1])
        
        loss_dis_real = (-torch.log(torch.sigmoid(real_out) + 1e-8)).mean()
        loss_dis_fake = (-torch.log(1 - torch.sigmoid(fake_out) + 1e-8)).mean()
        # print(loss_dis_real)
        # print(loss_dis_fake)
        loss_dis = loss_dis_real + loss_dis_fake
        
        
        loss_gen = (-torch.log(torch.sigmoid(fake_out) + 1e-8)).mean()
        if mode == 'all':
            return loss_gen, loss_dis
        elif mode == 'gen':
            return loss_gen
        else:
            return loss_dis
    return l

criterion = nn.BCELoss() 

def loss_gan_mnist():

    def l(weights, real_data, z, mode):
        gen_shapes = [(32, 64), (64, 32), (128, 64), (256, 128)]
        dis_shapes = [(128, 256), (64, 128), (32, 64), (1, 32)]


        weights_gen_split = []
        cur_sz = 0
        for shape in gen_shapes:
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]

        weights_dis_split = []
        for shape in dis_shapes:
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]    
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]
        
        z = F.linear(z, weights_gen_split[0], weights_gen_split[1])

        for i in range(2, len(weights_gen_split), 2):
            z = F.relu(z)
            z = F.linear(z, weights_gen_split[i], weights_gen_split[i + 1])
        z = F.tanh(z)
        fake_out = F.linear(z, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            fake_out = F.leaky_relu(fake_out, 0.2)
            # fake_out = F.dropout(fake_out, 0.3)
            fake_out = F.linear(fake_out, weights_dis_split[i], weights_dis_split[i + 1])
        
        fake_out = torch.sigmoid(fake_out)
        real_out = F.linear(real_data, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            real_out = F.leaky_relu(real_out, 0.2)
            # real_out = F.dropout(real_out, 0.3)
            real_out = F.linear(real_out, weights_dis_split[i], weights_dis_split[i + 1])
        real_out = torch.sigmoid(real_out)
        y_real = torch.ones(real_out.shape[0], 1).to(real_out.device)
        y_fake = torch.zeros(real_out.shape[0], 1).to(real_out.device)

        
        # print(loss_dis_real)
        # print(loss_dis_fake)
        loss_dis = criterion(real_out, y_real) + criterion(fake_out, y_fake)
        
        
        loss_gen = criterion(fake_out, y_real)
        if mode == 'all':
            return loss_gen, loss_dis
        elif mode == 'gen':
            return loss_gen
        else:
            return loss_dis
    return l



def loss_gan_high_dimension():

    def l(weights, real_data, z, mode):
        gen_shapes = [(200, 200), (200, 200), (75, 200)]
        dis_shapes = [(200, 75), (200, 200), (1, 200)]

        weights_gen_split = []
        cur_sz = 0
        for shape in gen_shapes:
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]
            weights_gen_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]

        weights_dis_split = []
        for shape in dis_shapes:
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0] * shape[1]].view(shape))
            cur_sz += shape[0] * shape[1]    
            weights_dis_split.append(weights[cur_sz:cur_sz + shape[0]].view(shape[0]))
            cur_sz += shape[0]
        
        z = F.linear(z, weights_gen_split[0], weights_gen_split[1])

        for i in range(2, len(weights_gen_split), 2):
            z = F.relu(z)
            z = F.linear(z, weights_gen_split[i], weights_gen_split[i + 1])

        fake_out = F.linear(z, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            fake_out = F.relu(fake_out)
            fake_out = F.linear(fake_out, weights_dis_split[i], weights_dis_split[i + 1])
        
        
        real_out = F.linear(real_data, weights_dis_split[0], weights_dis_split[1])
        for i in range(2, len(weights_dis_split), 2):
            real_out = F.relu(real_out)
            real_out = F.linear(real_out, weights_dis_split[i], weights_dis_split[i + 1])
        
        loss_dis_real = (-torch.log(torch.sigmoid(real_out) + 1e-8)).mean()
        loss_dis_fake = (-torch.log(1 - torch.sigmoid(fake_out) + 1e-8)).mean()
        # print(loss_dis_real)
        # print(loss_dis_fake)
        loss_dis = loss_dis_real + loss_dis_fake
        
        
        loss_gen = (-torch.log(torch.sigmoid(fake_out) + 1e-8)).mean()
        if mode == 'all':
            return loss_gen, loss_dis
        elif mode == 'gen':
            return loss_gen
        else:
            return loss_dis
    return l