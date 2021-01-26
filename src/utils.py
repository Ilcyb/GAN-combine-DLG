from torchvision import transforms
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
import pickle
import json
import math

tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()
toTensor = transforms.ToTensor()
gtp = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def labels_to_onehot(targets, num_classes=100):
    onehot_targets = torch.zeros(len(targets), targets[0].size(0), num_classes, device=targets[0].device)
    for idx in range(len(targets)):
        onehot_targets[idx] = label_to_onehot(targets[idx], num_classes)
    return onehot_targets

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def change_learning_rate(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr 

def calculate_psnr(img1, img2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

def get_save_path(training_num, config):
    if not os.path.isdir(config['dir']):
        os.makedirs(config['dir'])
    save_dir = 'ds-{}_bs-{}_init-{}_iter-{}_op-{}_nm-{}'.format(config['dataset'], 
                                        #    config['participants'],
                                           config['batch_size'],
                                           config['init_method'],
                                           config['iters'],
                                           config['optim'],
                                           config['norm_method'])
    if config['norm_method'] != 'none':
        save_dir += '_nr={}'.format(config['norm_rate'])
    if config['optim'] != 'LBFGS':
        save_dir += '_lr={}'.format(config['lr'])

    save_dir = os.path.join(config['dir'], os.path.join(save_dir, 'training_'+str(training_num)))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir

def save_checkpoint(mode, ckpt_location, exp_context):
    if mode == 'production':
        with open(ckpt_location, 'wb') as f:
            pickle.dump(exp_context, f, pickle.HIGHEST_PROTOCOL)
    else:
        return

def load_checkpoint(ckpt_location):
    try:
        with open(ckpt_location, 'rb') as f:
            exp_context = pickle.load(f)
    except FileNotFoundError as e:
        return None
    return exp_context

import torchvision.utils as vutils
def save_tensor_img(save_dir, filename, tensor, grid=False, grid_option=None):
    if not grid:
        vutils.save_image(tensor, os.path.join(save_dir, filename+'.png'))
    else:
        if grid_option is None:
            vutils.save_image(vutils.make_grid(tensor), 
                            os.path.join(save_dir, filename+'.png'))
        else:
            vutils.save_image(vutils.make_grid(tensor, **grid_option),
                            os.path.join(save_dir, filename+'.png'))

def save_plt_img(save_dir, filename):
    filename = os.path.join(save_dir, filename+'.png')
    plt.savefig(filename)

def read_experiment_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print('{} not exists'.format(config_path))
        exit(-1)

def get_truth_label(gradients):
    for i in range(1, len(gradients)-1):
        if gradients[i] * gradients[i-1] <=0 and gradients[i] * gradients[i+1] <= 0:
            return i
    
    if gradients[0] * gradients[1] <= 0:
        return 0

    if gradients[-1] * gradients[-2] <= 0:
        return len(gradients) - 1
    
    raise ValueError('{} 中没有符号与其他项不一致的项'.format(gradients))

def compute_smooth_by_martix(img_tensor):
    size = img_tensor.size()
    total_value = 0
    for channel in range(size[0]):
        m_left = torch.roll(img_tensor[channel], -1, 1)
        m_left_up = torch.roll(m_left, -1, 0)
        m_left_down = torch.roll(m_left, 1, 0)
        m_right = torch.roll(img_tensor[channel], 1, 1)
        m_right_up = torch.roll(m_right, -1, 0)
        m_right_down = torch.roll(m_right, 1, 0)
        m_up = torch.roll(img_tensor[channel], -1, 0)
        m_down = torch.roll(img_tensor[channel], 1, 0)
        
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_left))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_right))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_up))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_down))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_left_up))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_left_down))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_right_up))
        total_value += torch.sum(torch.abs(img_tensor[channel]-m_right_down))

    return total_value / (size[0]*size[1]*size[2])

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def check_folder_path(folder_paths):
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

def check_early_stop(psnr_list, early_stop_step_threshold=10, early_stop_var_threshold=10e-6):
    if len(psnr_list) < early_stop_step_threshold:
        return False
    var = torch.var(torch.Tensor(psnr_list[-1*early_stop_step_threshold:])).item()
    if var < early_stop_var_threshold:
        return True
    return False

def early_stop(psnrs, config, iter_num):
    if check_early_stop(psnrs, early_stop_step_threshold=config['early_stop_step']):
        config['early_stop'] = True
        config['iters'] = iter_num
        config['step_size'] = 1 if iter_num <= 100 else math.ceil(iter_num / 100)
        return True
    return False