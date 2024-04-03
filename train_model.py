import torch.nn as nn 
import sys, json, os, argparse
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from get_model import get_arch
from diffusion_framework import *
from utils.get_loaders import get_train_loaders

from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pynvml


# argument parsing
parser = argparse.ArgumentParser()

# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--csv_train', type=str, default='data/DRIVE/train.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='unet', help='architecture')
parser.add_argument('--batch_size', type=int, default=3, help='batch Size')
parser.add_argument('--grad_acc_steps', type=int, default=0, help='gradient accumulation steps (0)')
parser.add_argument('--min_lr', type=float, default=1e-8, help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch_num', type=int, default=1000, help='')
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')


def query_by_id(gpu_id):
    
 
    pynvml.nvmlInit()
    print(pynvml.nvmlDeviceGetCount())  
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
 
    #print(meminfo.total)  
    #print(meminfo.used)  
    #print(meminfo.free) 
    return meminfo.free / 1024 / 1024



def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None, mode=1,
        grad_acc_steps=0, assess=False):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    if assess: logits_all, labels_all = [], []
    
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            BATCH_SIZE = labels.size(0)
            if mode==1:
                time_step = torch.randint(1, T//10, (BATCH_SIZE,), device=device).long()
            elif mode==2:
                time_step = torch.randint(1, T, (BATCH_SIZE,), device=device).long()
            

            if mode==1:
                logits, noise_pred = model(inputs, inputs, time_step.double())
                #print(logits.size())
                loss = criterion(logits, labels.unsqueeze(dim=1).float()) 
            elif mode==2:
                x_noisy, noise = forward_diffusion_sample_last(inputs, time_step, device)
                logits, noise_pred = model(inputs, x_noisy, time_step.double())
                loss = F.l1_loss(noise, noise_pred)
            
            
            if train:  # only in training mode
                (loss / (grad_acc_steps + 1)).backward() # for grad_acc_steps=0, this is just loss
                if i_batch % (grad_acc_steps+1) == 0:  # for grad_acc_steps=0, this is always True
                    optimizer.step()
                    for _ in range(grad_acc_steps+1): scheduler.step() # for grad_acc_steps=0, this means once
                    optimizer.zero_grad()
            

            # Compute running loss
            running_loss += loss.item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(run_loss), get_lr(optimizer)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    if assess: return run_loss
    return None, None, None

def train_model(model, optimizer1, optimizer2, criterion, train_loader, scheduler1, scheduler2, grad_acc_steps, exp_path):

    cycle_len = scheduler1.epoch_num
    
    # reset iteration counter
    scheduler1.last_epoch = -1
    scheduler2.last_epoch = -1
    # update number of iterations
    scheduler1.T_max = scheduler1.epoch_num * len(train_loader)//20
    scheduler2.T_max = scheduler2.epoch_num * len(train_loader)//20
    for epoch in range(cycle_len):  #cycle_len
        print('Epoch {:d}/{:d}'.format(epoch+1, cycle_len))
        optimizer2.zero_grad()
        # prepare next cycle:
        
        assess = False
        tr_loss = run_one_epoch(train_loader, model, criterion, optimizer=optimizer2,
                                                      scheduler=scheduler2, mode=2, grad_acc_steps=grad_acc_steps, assess=assess)
        
    for epoch in range(cycle_len):
        print('Epoch {:d}/{:d}'.format(epoch+1, cycle_len))
        optimizer1.zero_grad()
        # prepare next cycle:
        if epoch%10 == 0:
            assess = True
        else:
            assess = False
        
        tr_loss = run_one_epoch(train_loader, model, criterion, optimizer=optimizer1,
                                                      scheduler=scheduler1, mode=1, grad_acc_steps=grad_acc_steps, assess=assess)
        
        if assess == True:
            loss = tr_loss
            print('save model')
            if exp_path is not None:
                print(15 * '-', ' Checkpointing ', 15 * '-')
                save_model(exp_path, model, optimizer1)

    del model
    torch.cuda.empty_cache()
    return loss

if __name__ == '__main__':
    '''
    Example:
    python train_cyclical.py --csv_train data/DRIVE/train.csv --save_path unet_DRIVE
    '''

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Training on device '{args.device}'...")
        device = torch.device("cuda")

    else:  #cpu
        device = torch.device(args.device)

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, args.device.startswith("cuda"))

    # gather parser parameters
    model_name = args.model_name
    max_lr, min_lr, bs, grad_acc_steps = args.max_lr, args.min_lr, args.batch_size, args.grad_acc_steps
    epoch_num = args.epoch_num

    

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path=None

    csv_train = args.csv_train
    
    n_classes=1
    label_values = [0, 255]

    print(f"* Creating Dataloaders, batch size = {bs}, workers = {args.num_workers}")
    train_loader = get_train_loaders(csv_path_train=csv_train, batch_size=bs, tg_size=tg_size, label_values=label_values, num_workers=args.num_workers)


    fl=0;
    while fl==0:
        memory_free = query_by_id(int(args.device.split(":",1)[1]))
        print("gpu %d free memory is %d M." % (int(args.device.split(":",1)[1]), memory_free))
        if memory_free>20000:
            fl=1
    print('* Instantiating a {} model'.format(model_name))
    model = get_arch(model_name, n_classes=n_classes)
    model = model.to(device)

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer1 = torch.optim.Adam([{'params':model.encoder.parameters()}, {'params':model.trans.parameters()}, {'params':model.decoder.parameters()}], lr=max_lr)
    optimizer2 = torch.optim.Adam([{'params':model.encoder.parameters()}, {'params':model.difencoder.parameters()}, {'params':model.trans.parameters()}, {'params':model.difdecoder.parameters()}], lr=max_lr)

    scheduler1 = CosineAnnealingLR(optimizer1, T_max=epoch_num * len(train_loader) // (grad_acc_steps + 1), eta_min=min_lr)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=epoch_num * len(train_loader) // (grad_acc_steps + 1), eta_min=min_lr)
    setattr(optimizer1, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(optimizer2, 'max_lr', max_lr)
    setattr(scheduler1, 'epoch_num', epoch_num)
    setattr(scheduler2, 'epoch_num', epoch_num)

    criterion = torch.nn.BCEWithLogitsLoss()

    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n','-' * 10)

    m1=train_model(model, optimizer1, optimizer2, criterion, train_loader, scheduler1, scheduler2, grad_acc_steps, experiment_path)

    print("train_loss: %f" % m1)
    
