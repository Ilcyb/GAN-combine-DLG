from pprint import pprint
import os
import math
import time
import argparse
import sys
import pathlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import torch
from torchvision import models, datasets, transforms

sys.path.append('..')

from utils import *  # noqa: E402
from models import *  # noqa: E402

plt.rcParams.update({'figure.max_open_warning': 0})
torch.manual_seed(50)

def get_real_datas(net, save_dir, config):
    dataset = config['dataset']
    dst = None
    if dataset == 'cifar10':
        dst = datasets.CIFAR10(base_config['dataset']['cifar10_path'], download=True)
    elif dataset == 'mnist':
        dst = datasets.MNIST(base_config['dataset']['mnist_path'], download=True)
    elif dataset == 'cifar100':
        dst = datasets.CIFAR100(base_config['dataset']['cifar100_path'], download=True)
    elif dataset == 'svhn':
        dst = datasets.SVHN(base_config['dataset']['svhn_path'], download=True)

    participants = config['participants']
    batch_size = config['batch_size']
    img_idxs = [np.random.choice(range(10000), batch_size) for i in range(participants)]
    leak_data_size = tp(dst[0][0]).size()

    data_shape = (participants, batch_size, *leak_data_size)
    label_shape = (participants, batch_size)
    gt_data = torch.randn(data_shape).to(device)
    gt_label = torch.zeros(label_shape).long().to(device)
    gt_onehot_label = []
    for i in range(participants):
        labels = []
        for j in range(batch_size):
            gt_data[i][j] = tp(dst[img_idxs[i][j]][0]).to(device)
            labels.append(dst[img_idxs[i][j]][1])
        gt_label[i] = torch.Tensor(labels).long().to(device)
        gt_onehot_label.append(label_to_onehot(gt_label[i], num_classes=config['num_classes']))
    label_onehot_shape = (participants, *gt_onehot_label[0].size())

    # compute original gradient 
    total_dy_dx = []
    for i in range(participants):
        out = net(gt_data[i])
        y = criterion(out, gt_onehot_label[i])
        dy_dx = torch.autograd.grad(y, net.parameters())

        # FIXME 
        # 这里因为实现的问题直接用真实标签代替了iDLG算法算出来的标签
        # 相当于默认iDLG算法绝对正确
        # 但其实不能这么做
        # truth_label = get_truth_label(dy_dx[-1])
        # gt_onehot_label[i] = label_to_onehot(torch.Tensor([truth_label]).long().to(device), num_classes=config['num_classes'])

        # share the gradients with other clients
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        for j in range(len(original_dy_dx)):
            if len(total_dy_dx) <= j:
                total_dy_dx.append(original_dy_dx[j])
            else:
                total_dy_dx[j] += original_dy_dx[j]

    mean_dy_dx = []
    for i in range(len(total_dy_dx)):
        mean_dy_dx.append(total_dy_dx[i] / participants)

    for i in range(participants):
        for j in range(batch_size):
            # plt.figure()
            # plt.imshow(tt(gt_data[i][j].cpu()))
            # plt.title("Ground truth image; Participant is {};GT label is {};GT onehot label is {}".format(
            #     i+1, gt_label[i][j], torch.argmax(gt_onehot_label[i][j], 0)))
            # save_plt_img(save_dir, 'truth_img{}-{}'.format(i+1, j+1))
            save_tensor_img(save_dir, 'truth_img{}-{}'.format(i + 1, j + 1), gt_data[i][j].cpu())
        save_tensor_img(save_dir, 'truth_img_grid', gt_data[i], True)

    return gt_data, gt_onehot_label, mean_dy_dx, data_shape, label_onehot_shape


def cpl_patterned(data_shape):
    # data_shape:[batch_size, channel, width, height]
    if data_shape[2] % 2 != 0 or data_shape[3] % 2 != 0:
        raise ValueError('[{}*{}] cannot cpl patterned'.format(data_shape[2], data_shape[3]))
    dummy = torch.zeros(data_shape)
    channel_templates = []
    template_w = (int)(data_shape[2] / 2)
    template_h = (int)(data_shape[3] / 2)
    for c in range(data_shape[1]):
        channel_templates.append(torch.rand(template_w * template_h))
    for bs in range(data_shape[0]):
        for c in range(data_shape[1]):
            count = 0
            for i in range(template_w):
                for j in range(template_h):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w, data_shape[2]):
                for j in range(template_h):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w):
                for j in range(template_h, data_shape[3]):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w, data_shape[2]):
                for j in range(template_h, data_shape[3]):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1
    return dummy.to(device).requires_grad_(True)


def cpl_rgb(data_shape):
    return torch.ones(data_shape).to(device).requires_grad_(True)


def cpl_dark(data_shape):
    return torch.zeros(data_shape).to(device).requires_grad_(True)


def gan_dummy(data_shape, participant, batch_size, onehot_labels, config, generate_models):
    dummies = []
    for i in range(batch_size):
        label = int(torch.argmax(onehot_labels[participant][i].squeeze()).item())
        dummy_data = gtp(tt(generate_models[label](torch.randn(1, 100, 1, 1))[0])).view(1, *(data_shape[2:]))
        dummies.append(dummy_data)
    return torch.cat(dummies, 0).to(device).requires_grad_(True)


def pure_gan_dummy(data_shape, participant, batch_size, config, generate_model):
    dummies = []
    for i in range(batch_size):
        if config['dataset'] == 'mnist':
            # z = Variable(Tensor(np.random.normal(0, 1, (1, 100))))
            z = torch.randn(1, 100)
        elif config['dataset'] == 'cifar10':
            z = torch.randn(1, 100, 1, 1, device=device)
        elif config['dataset'] == 'svhn':
            z = torch.randn((1, 100), device=device)
        dummy_data = gtp(tt(generate_model(z).squeeze())).view(1, *(data_shape[2:]))
        dummies.append(dummy_data)
    return torch.cat(dummies, 0).to(device).requires_grad_(True)


# generate dummy data and label
def generate_dummy_datas(save_dir, config, data_shape, label_onehot_shape, 
                        generate_model, generate_models, gt_onehot_labels=None):
    dummy_datas = []
    dummy_labels = []
    participants = config['participants']
    batch_size = config['batch_size']

    for i in range(participants):
        if config['init_method'] == 'cpl-rgb':
            dummy_datas.append(cpl_rgb(data_shape[1:]))
            dummy_labels.append(cpl_rgb(label_onehot_shape[1:]))
        elif config['init_method'] == 'cpl-dark':
            dummy_datas.append(cpl_rgb(data_shape[1:]))
            dummy_labels.append(cpl_rgb(label_onehot_shape[1:]))
        elif config['init_method'] == 'cpl-patterned':
            dummy_datas.append(cpl_patterned(data_shape[1:]))
            dummy_labels.append(torch.randn(label_onehot_shape[1:]).to(device).requires_grad_(True))
        elif config['init_method'] == 'gan':
            dummy_datas.append(gan_dummy(data_shape, i, batch_size, gt_onehot_labels, config, generate_models))
            dummy_labels.append(gt_onehot_labels[i].detach().to(device).requires_grad_(True))
        elif config['init_method'] == 'pure-gan':
            dummy_datas.append(pure_gan_dummy(data_shape, i, batch_size, config, generate_model))
            dummy_labels.append(gt_onehot_labels[i].detach().to(device).requires_grad_(True))
        else:
            dummy_datas.append(torch.rand(data_shape[1:]).to(device).requires_grad_(True))
            dummy_labels.append(torch.randn(label_onehot_shape[1:]).to(device).requires_grad_(True))

    # dummy_data = torch.randn((batch_size, 1, *gt_data.size()[1:])).to(device).requires_grad_(True)
    # dummy_label = torch.randn((batch_size, 1, *gt_onehot_label.size()[1:])).to(device).requires_grad_(True)

    for i in range(participants):
        for j in range(batch_size):
            # plt.figure()
            # plt.imshow(tt(dummy_datas[i][j].cpu()))
            # plt.title("Participant {}; Dummy data {}; Dummy label is {}".format(
            #     i+1, j+1, torch.argmax(dummy_labels[i][j], 0)))
            # save_plt_img(save_dir, 'dummy_img{}-{}'.format(i+1, j+1))
            save_tensor_img(save_dir, 'dummy_img{}-{}'.format(i + 1, j + 1), dummy_datas[i][j].cpu())
        save_tensor_img(save_dir, 'dummy_img_grid', dummy_datas[i], True)

    return dummy_datas, dummy_labels


# optimizer = torch.optim.LBFGS([dummy_data, dummy_label] )
def recover(save_dir, config, net, gt_data, dummy_datas, dummy_labels, mean_dy_dx):
    dummies = []
    history = []
    loss = []
    psnrs = []

    participants = config['participants'] or 1
    batch_size = config['batch_size'] or 1
    iters = config['iters'] or 10000
    step_size = config['step_size'] or 1000
    lr = config['lr'] or 0.02
    optim = config['optim'] or 'adam'
    norm_rate = config['norm_rate']

    for i in range(participants):
        dummies.append(dummy_datas[i])
        dummies.append(dummy_labels[i])
        _ = []
        for j in range(batch_size):
            _.append([])
        history.append(_)
    # optimizer = torch.optim.Adam(dummies, lr=lr)
    optimizer = torch.optim.LBFGS(dummies)

    for i in range(participants):
        for j in range(batch_size):
            # 将初始化噪声加入history
            history[i][j].append(dummy_datas[i][j].cpu().clone())

    start_time = time.time()
    for iter in range(iters):
        def closure():
            # compute mean dummy dy/dx
            total_dy_dx = []
            optimizer.zero_grad()
            smooth = 0
            for i in range(participants):
                pred = net(dummy_datas[i])
                dummy_loss = criterion(pred, dummy_labels[i])
                dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                # share the gradients with other clients
                dummy_dy_dx = [_ for _ in dy_dx]
                for j in range(len(dummy_dy_dx)):
                    if len(total_dy_dx) <= j:
                        total_dy_dx.append(dummy_dy_dx[j])
                    else:
                        total_dy_dx[j] += dummy_dy_dx[j]

            dummy_mean_dy_dx = []
            for i in range(len(total_dy_dx)):
                dummy_mean_dy_dx.append(total_dy_dx[i] / participants)

            grad_diff = 0
            for gx, gy in zip(dummy_mean_dy_dx, mean_dy_dx):  # TODO: fix the variablas here
                grad_diff += ((gx - gy) ** 2).sum()

            if config['norm_method'] == 'smooth':
                # 图片smooth程度正则项
                for i in range(participants):
                    for j in range(batch_size):
                        smooth += compute_smooth_by_martix(dummy_datas[i][j])
                        # print(smooth)
                grad_diff += norm_rate * smooth

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure()
        loss.append(current_loss.item())

        mean_psnr = calculate_psnr(dummy_datas[0].cpu().clone().detach(), gt_data[0].cpu().clone().detach())
        psnrs.append(mean_psnr)
        # mean_psnr = 0
        # for i in range(participants):
        #     for j in range(batch_size):
        #         psnr = calculate_psnr(tt(dummy_datas[i][j].cpu()), tt(gt_data[i][j].cpu()))
        #         mean_psnr += psnr
        # mean_psnr = mean_psnr / (participants*batch_size)
        # psnrs.append(mean_psnr)

        if (iter % step_size == 0) or iter == iters - 1:
            for i in range(participants):
                for j in range(batch_size):
                    history[i][j].append(dummy_datas[i][j].cpu().clone())
            print("iter:{}\tloss:{:.5f}\tmean_psnr:{:.5f}\tcost time:{:.2f} secs".format(iter, current_loss.item(), mean_psnr, time.time() - start_time))
            start_time = time.time()

    for i in range(participants):
        for j in range(batch_size):
            history[i][j].append(dummy_datas[i][j].cpu().clone())

    return dummy_datas, dummy_labels, history, loss[1:], psnrs


def create_plt(save_dir, config, gt_data, dummy_datas, dummy_labels, history, loss, psnrs=None):
    participants = config['participants']
    batch_size = config['batch_size']
    iters = config['iters']
    step_size = config['step_size']

    row = math.ceil(iters / step_size / 10)
    compare_result = []
    compare_truth = []
    history_grid = []
    for p in range(participants):
        for j in range(batch_size):
            # plt.figure()
            # plt.imshow(tt(dummy_datas[p][j].cpu()))
            # save_plt_img(save_dir, 'result{}-{}'.format(p+1, j+1))
            history_grid = []
            save_tensor_img(save_dir, 'result{}-{}'.format(p + 1, j + 1), dummy_datas[p][j].cpu())
            plt.figure(figsize=(20, 6))
            print("Participant {} Dummy label {} is {}.".format(p + 1, j + 1,
                                                                torch.argmax(dummy_labels[p][j], dim=-1).item()))
            for i in range((int)(iters / step_size)):
                # plt.subplot(row, 10, i + 1)
                # plt.title("iter=%d" % ((i)*step_size))
                # plt.imshow(history[p][j][i])
                # plt.axis('off')
                history_grid.append(history[p][j][i])
            history_grid.append(history[p][j][-1])
            # save_plt_img(save_dir, 'procedure{}-{}'.format(p+1, j+1))
            compare_result.append(dummy_datas[p][j])
            compare_truth.append(gt_data[p][j])
            save_tensor_img(save_dir, 'procedure{}-{}'.format(p + 1, j + 1),
                            history_grid, True,
                            dict(nrow=math.ceil(math.sqrt(len(history_grid)))))
        save_tensor_img(save_dir, 'result_img_grid', dummy_datas[p], True)
        save_tensor_img(save_dir, 'compare',
                        [vutils.make_grid(dummy_datas[p], len(dummy_datas[p])),
                         vutils.make_grid(gt_data[p], len(gt_data[p]))],
                        True, dict(nrow=1))

    plt.figure(figsize=(12, 6))
    loss_ = [loss[i] for i in range(len(loss)) if i % 1 == 0]
    x = [i + 1 for i in range(len(loss)) if i % 1 == 0]
    plt.plot(x, loss_, color='#000000', label='loss')
    plt.title('Loss(log)')
    plt.xlabel('iter')
    plt.yscale('log')
    plt.ylabel('loss value')
    plt.xticks(range(len(x)))
    x_major_locator = MultipleLocator((int)(config['iters'] / 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    save_plt_img(save_dir, 'loss')

    plt.figure(figsize=(12, 6))
    psnrs_ = [psnrs[i] for i in range(len(psnrs)) if i % 1 == 0]
    x = [i + 1 for i in range(len(psnrs)) if i % 1 == 0]
    plt.plot(x, psnrs_, color='#000000', label='psnr')
    plt.title('Mean Psnr')
    plt.xlabel('iter')
    plt.ylabel('mean psnr value')
    plt.xticks(range(len(x)))
    x_major_locator = MultipleLocator((int)(config['iters'] / 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    save_plt_img(save_dir, 'meanpsnr')

    loss_log_path = os.path.join(save_dir, 'loss.log')
    loss_str_list = ['iter-{}:{:.5f}'.format(i + 1, loss[i]) for i in range(len(loss))]
    with open(loss_log_path, 'w') as f:
        f.write('\n'.join(loss_str_list))

    psnr_log_path = os.path.join(save_dir, 'meanpsnr.log')
    psnr_str_list = ['iter-{}:{:.5f}'.format(i + 1, psnrs[i]) for i in range(len(psnrs))]
    with open(psnr_log_path, 'w') as f:
        f.write('\n'.join(psnr_str_list))
    

def experiment_config_loop(mode, ckpt_location, context, experiments, current_config_idx, config_name):
    experiments[current_config_idx] = (experiments[current_config_idx] + 1) % len(
        experiments[config_name])
    context['experiments'] = experiments
    save_checkpoint(mode, ckpt_location, context)

def experiment(mode, device, experiments, iters, config, base_generate_model_path, **idx):
    print('''
========================================================
Mode: {}
Batch Size: {}
Training Num: {}
Optimizer: {}
Learning Rate: {}
Init Method: {}
Dataset: {}
Norm Method: {}
Norm Rate: {}
Iters: {}
    '''.format(
        mode,
        experiments['batch_size'][idx['b_idx']],
        experiments['training_num'][idx['t_idx']],
        experiments['optim'][idx['o_idx']],
        experiments['lr'][idx['l_idx']],
        experiments['init'][idx['init_idx']],
        experiments['dataset'][idx['ds_idx']].upper(),
        experiments['norm_methods'][idx['nm_idx']],
        experiments['norm_rate'][idx['nr_idx']],
        iters
    ))

    start_time = time.time()
    config['batch_size'] = experiments['batch_size'][idx['b_idx']]
    config['lr'] = experiments['lr'][idx['l_idx']]
    config['optim'] = experiments['optim'][idx['o_idx']]
    config['init_method'] = experiments['init'][idx['init_idx']]
    config['dataset'] = experiments['dataset'][idx['ds_idx']]
    config['norm_rate'] = experiments['norm_rate'][idx['nr_idx']]
    config['norm_method'] = experiments['norm_methods'][idx['nm_idx']]

    generate_models = []
    generate_model = None
    if config['init_method'] == 'gan' and config['dataset'] == 'mnist':
        generate_model_path = os.path.join(base_generate_model_path,
                                            'MNIST-GenerateModel')
        for j in range(10):
            m = DCGANGenerator_mnist()
            m.load_state_dict(torch.load(os.path.join(generate_model_path,
                                                        'generate_model_{}'.format(
                                                            j))))
            m.eval()
            generate_models.append(m)
    elif config['init_method'] == 'gan' and config['dataset'] == 'cifar10':
        generate_model_path = os.path.join(base_generate_model_path,
                                            'CIFAR10-GenerateModel')
        pass
    elif config['init_method'] == 'pure-gan' and config['dataset'] == 'mnist':
        generate_model_path = os.path.join(base_generate_model_path,
                                            'MNIST-PureGenerateModel')
        from torch.autograd import Variable

        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        generate_model = PureMnistGenerator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'generator-state-epoch-100'),
            map_location=torch.device(device)))
        generate_model.eval()
    elif config['init_method'] == 'pure-gan' and config['dataset'] == 'cifar10':
        generate_model_path = os.path.join(base_generate_model_path,
                                            'CIFAR10-PureGenerateModel')
        generate_model = PureCifar10Generator()
        generate_model.load_state_dict(
            torch.load(os.path.join(generate_model_path, 'netG_epoch_24.pth'),
                        map_location=torch.device(device)))
        generate_model.eval()
    elif config['init_method'] == 'pure-gan' and config['dataset'] == 'svhn':
        generate_model_path = os.path.join(base_generate_model_path,
                                            'SVHN-PureGenerateModel')
        generate_model = SVHNPureGenerator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'SVHN_Generator_epoch50.mdl'),
            map_location=torch.device(device)))
        generate_model.eval()

    net = None
    if config['dataset'] == 'mnist':
        config['num_classes'] = 10
        net = LeNet_Mnist().to(device)
    elif config['dataset'] == 'cifar10':
        config['num_classes'] = 10
        net = LeNet_Cifar10().to(device)
    elif config['dataset'] == 'cifar100':
        config['num_classes'] = 100
        net = LeNet_Cifar100().to(device)
    elif config['dataset'] == 'svhn':
        config['num_classes'] = 10
        net = LeNet_SVHN().to(device)
    net.apply(weights_init)

    save_dir = get_save_path(experiments['training_num'][idx['t_idx']], config)
    gt_data, recoverd_onehot_label, mean_dy_dx, data_shape, label_onehot_shape = get_real_datas(
                                                                            net, save_dir, config)
    dummy_datas, dummy_labels = generate_dummy_datas(save_dir, config,
                                                        data_shape,
                                                        label_onehot_shape,
                                                        generate_model,
                                                        generate_models,
                                                        recoverd_onehot_label)
    dummy_datas, dummy_labels, history, loss, psnrs = recover(save_dir, config,
                                                                net, gt_data, dummy_datas,
                                                                recoverd_onehot_label,
                                                                mean_dy_dx)
    create_plt(save_dir, config, gt_data, dummy_datas, dummy_labels, history,
                loss, psnrs)
    plt.close('all')
    cost_time = time.time() - start_time
    print('\ntime cost: {} secs'.format(cost_time))
    print('========================================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN combine DLG')
    parser.add_argument('--config-file', type=str, help='experiment config file path')
    parser.add_argument('--mode', type=str, default='debug', help='experiment mode')
    parser.add_argument('--device', type=str, default='cuda', help='running device')
    parser.add_argument('--base-config', type=str, default='../base_config.json')
    args = parser.parse_args()

    # base_config = read_experiment_config(os.path.join(pathlib.Path(__file__).parent.absolute().parent, 'base_config.json'))
    base_config = read_experiment_config(args.base_config)
    folder_paths = [base_config['production_path'], base_config['debug_path'], base_config['generate_model_path']]
    check_folder_path(folder_paths)

    experiment_config = read_experiment_config(args.config_file)
    experiment_name = experiment_config['name']
    participants = experiment_config['participants']
    batch_size = experiment_config['batch_size']
    data_set = experiment_config['data_set']
    training_num = experiment_config['training_num']
    init_methods = experiment_config['init_methods']
    norm_methods = experiment_config['norm_methods']
    norm_rate = experiment_config['norm_rate']
    iters = experiment_config['iters']
    mode = args.mode

    if torch.cuda.is_available() and args.device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    print("Running on %s" % device)

    assert participants == [1]

    production_path = base_config['production_path']
    debug_path = base_config['debug_path']
    base_generate_model_path = base_config['generate_model_path']
    path = debug_path

    if mode == 'production':
        path = production_path
    path = os.path.join(path, experiment_name)

    config = {
        'participants': 1,
        'batch_size': 1,
        'dataset': 'mnist',
        'lr': 0.002,
        'optim': 'LBFGS',
        'iters': iters,
        'step_size': 1 if iters <= 100 else math.ceil(iters / 100),
        'dir': path,
        'init_method': 'gan',
        'norm_method': 'none',
        'norm_rate': 0.0001,
        'regular_ratio': 0
    }

    criterion = cross_entropy_for_onehot
    experiments = {
        'participants': participants,
        'current_participant': 0,
        'batch_size': batch_size,
        'current_bs': 0,
        'dataset': data_set,
        'current_ds': 0,
        'training_num': [i for i in range(1, training_num + 1)],
        'current_tn': 0,
        'optim': ['LBFGS'],
        'current_opt': 0,
        'lr': [0.005],
        'current_lr': 0,
        'norm_rate': norm_rate,
        'current_nr': 0,
        'init': init_methods,
        'current_init': 0,
        'norm_methods': norm_methods,
        'current_nm': 0
    }

    done = False
    ckpt_location = os.path.join(path, 'context.ckpt')
    context = load_checkpoint(ckpt_location)
    if context is None:
        context = {
            'done': False,
            'config': config,
            'experiments': experiments
        }
        print('init experiments context')
    elif context['done'] == False:
        config = context['config']
        experiments = context['experiments']
        print('load unfinished experiment')
        print(experiments)
    elif context['done'] == True:
        print('All experiments are done!')
        done = True

    if not done:
        current_participant = experiments['current_participant']
        current_bs = experiments['current_bs']
        current_tn = experiments['current_tn']
        current_opt = experiments['current_opt']
        current_lr = experiments['current_lr']
        current_init = experiments['current_init']
        current_ds = experiments['current_ds']
        current_nr = experiments['current_nr']
        current_nm = experiments['current_nm']

        for b_idx in range(current_bs, len(experiments['batch_size'])):
            for t_idx in range(current_tn, len(experiments['training_num'])):
                for o_idx in range(current_opt, len(experiments['optim'])):
                    for l_idx in range(current_lr, len(experiments['lr'])):
                        for init_idx in range(current_init, len(experiments['init'])):
                            for ds_idx in range(current_ds, len(experiments['dataset'])):
                                for nr_idx in range(current_nr, len(experiments['norm_rate'])):
                                    for nm_idx in range(current_nm, len(experiments['norm_methods'])):
                                        idx = dict(b_idx=b_idx, t_idx=t_idx, o_idx=o_idx,
                                        l_idx=l_idx, init_idx=init_idx, ds_idx=ds_idx, nr_idx=nr_idx,
                                        nm_idx=nm_idx)
                                        experiment(mode, device, experiments, iters, config, base_generate_model_path, **idx)
                                        experiment_config_loop(mode, ckpt_location, context, experiments, 'current_nm', 'norm_methods')
                                    experiment_config_loop(mode, ckpt_location, context, experiments, 'current_nr', 'norm_rate')
                                experiment_config_loop(mode, ckpt_location, context, experiments, 'current_ds', 'dataset')
                            experiment_config_loop(mode, ckpt_location, context, experiments, 'current_init', 'init')
                        experiment_config_loop(mode, ckpt_location, context, experiments, 'current_lr', 'lr')
                    experiment_config_loop(mode, ckpt_location, context, experiments, 'current_opt', 'optim')
                experiment_config_loop(mode, ckpt_location, context, experiments, 'current_tn', 'training_num')
            experiment_config_loop(mode, ckpt_location, context, experiments, 'current_bs', 'batch_size')

    context['done'] = True
    save_checkpoint(mode, ckpt_location, context)
    print('All Expriments Done!')
