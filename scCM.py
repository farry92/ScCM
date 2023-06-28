import argparse
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models
import bgi.loader
import bgi.model
from sklearn.cluster import KMeans
import scanpy as sc
import pandas as pd
from metrics import compute_metrics

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch scRNA-seq ScCM Training')

parser.add_argument('--input_h5ad_path', type=str,default= "",help='path to input h5ad file')
parser.add_argument('--obs_label_colname',type=str, default= None,help='column name of the label in obs')
parser.add_argument('-j','--workers',default=1,type=int,metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs',default=10,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--start_epoch',default=5,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b','--batch_size',default=512, type=int,metavar='N',help='mini-batch size (default: 256), this is the total ''batch size of all GPUs on the current node when ''using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr','--learning_rate',default=1e-6,type=float,metavar='LR',help='initial learning rate',dest='lr')
parser.add_argument('--wd','--weight_decay',default=1e-6,type=float,metavar='W',help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('--resume',default='',type=str,metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--schedule',default=[100, 120],nargs='*',type=int,help='learning rate schedule (when to drop lr by 10x), if use cos, then it will not be activated')
parser.add_argument('--low_dim',default=128,type=int,help='feature dimension (default: 128)')
parser.add_argument('--moco-dim', default=128, type=int,help='feature dimension (default: 128)')
parser.add_argument('--moco-mlp-dim', default=512, type=int,help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',help='gradually increase moco momentum to 1 with a ''half-cycle cosine schedule')
parser.add_argument('--moco-t', default=0.8, type=float,help='softmax temperature (default: 1.0)')
parser.add_argument('--cos', action='store_true',help='use cosine lr schedule')
parser.add_argument("--aug_prob",type=float,default=0,help="The prob of doing augmentation")
parser.add_argument('--cluster_name',default='kmeans', type=str,help='name of clustering method', dest="cluster_name")
parser.add_argument('--num_cluster', default=-1, type=int,help='number of clusters',dest="num_cluster")
parser.add_argument('--seed',default=35, type=int,help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,help='GPU id to use.')
parser.add_argument('-e','--eval_freq', default=1, type=int,metavar='N', help='Save frequency (default: 10)',dest='eval_freq')
parser.add_argument('-l', '--log_freq', default=10,type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--exp_dir',default='./experiment_pcl',type=str,help='experiment directory')
parser.add_argument('--save_dir',default='./results', type=str,help='result saving directory')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',help='number of warmup epochs')

def main():
    args = parser.parse_args()
    if args.seed is not None:        
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    main_worker(args)
    
def main_worker(args):
    end = time.time()
    start = end # Add this line to track the start time of training
    print(args)
    print("=> load data...")
    input_h5ad_path = args.input_h5ad_path
    processed_adata = sc.read_h5ad(input_h5ad_path) 
    obs_label_colname = args.obs_label_colname
    pre_path, filename = os.path.split(input_h5ad_path)
    dataset_name, ext = os.path.splitext(filename)
    if dataset_name == "counts":
        dataset_name = pre_path.split("/")[-1]
    if dataset_name == "":
        dataset_name = "unknown"
    save_path = os.path.join(args.save_dir, "SCCM")
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)

    args_transformation = {
        # crop
        # without resize, it's better to remove crop        
        # mask
        'mask_percentage': 0.25,
        'apply_mask_prob': 0,        
        # (Add) gaussian noise
        'noise_percentage': 0.8,
        'sigma': 0.5,
        'apply_noise_prob': 0,
        # swap with the NB regressed mu
        'nb_percentage': 0.8,
        'apply_nb_prob': 0,
        # inner swap
        'swap_percentage': 0.4,
        'apply_swap_prob': 0,        
        # cross over with 1
        'cross_percentage': 0.3,
        'apply_cross_prob': 0,        
        # cross over with many
        'change_percentage': 0.3,
        'apply_mutation_prob': 0
    }
     
    print("=> bulid loader...")    
    train_dataset = bgi.loader.scRNAMatrixInstance(
        adata=processed_adata,
        obs_label_colname=obs_label_colname,transform=True,
        args_transformation=args_transformation)        
    eval_dataset = bgi.loader.scRNAMatrixInstance(
        adata=processed_adata,
        obs_label_colname=obs_label_colname,transform=False)
    if train_dataset.num_cells < 512:
        args.batch_size = train_dataset.num_cells
    train_sampler = None
    eval_sampler = None        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=args.batch_size,shuffle=(train_sampler is None),
        num_workers=args.workers,pin_memory=True,sampler=train_sampler,drop_last=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,batch_size=args.batch_size * 5,shuffle=False,
        sampler=eval_sampler,num_workers=args.workers,pin_memory=True)

    print("=> creating model 'MLP'")
    model = bgi.model.MoCoV3(
        bgi.model.MLPEncoder,
        int(train_dataset.num_genes),
        args.moco_dim,
        args.moco_t)
    print(model)   
    args.lr = args.lr * args.batch_size / 256   
    if args.gpu is None:
        raise Exception("Should specify GPU id for training with --gpu".format(args.gpu))
    else:
        print("Use GPU: {} for training".format(args.gpu))
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)       
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    # optionally resume from a checkpoint 
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    for epoch in range(args.start_epoch, args.epochs):
        train_unsupervised_metrics = train(train_loader, model, optimizer, scaler, epoch, args)
        if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
            if epoch == 0:
                with open(os.path.join(save_path, 'log_scCM_{}.txt'.format(dataset_name)), "w") as f:
                    f.writelines(f"epoch\t" + '\t'.join((str(key) for key in train_unsupervised_metrics.keys())) + "\n")
                    f.writelines(f"{epoch}\t" + '\t'.join((str(train_unsupervised_metrics[key]) for key in train_unsupervised_metrics.keys())) + "\n")
            else:
                with open(os.path.join(save_path, 'log_scCM_{}.txt'.format(dataset_name)), "a") as f:
                    f.writelines(f"{epoch}\t" + '\t'.join((str(train_unsupervised_metrics[key]) for key in train_unsupervised_metrics.keys())) + "\n")

        # inference log & supervised metrics
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            embeddings, gt_labels = inference(eval_loader, model)

            # perform kmeans
            if args.cluster_name == "kmeans":
                # if gt_label exists and metric can be computed
                if train_dataset.label is not None:
                    num_cluster = len(train_dataset.unique_label) if args.num_cluster == -1 else args.num_cluster
                    print("cluster num is set to {}".format(num_cluster))
                    # multiple random experiments
                    best_ari, best_eval_supervised_metrics, best_pd_labels = -1, None, None
                    for random_seed in range(1):
                        pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embeddings).labels_
                        # compute metrics
                        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
                        if eval_supervised_metrics["ARI"] > best_ari:
                            best_ari = eval_supervised_metrics["ARI"]
                            best_eval_supervised_metrics = eval_supervised_metrics
                            best_pd_labels = pd_labels
                    print("Epoch: {}\t {}\n".format(epoch, eval_supervised_metrics))
                    with open(os.path.join(save_path, 'log_ScCM_{}.txt'.format(dataset_name)), "a") as f:
                        f.writelines("{}\teval\t{}\n".format(epoch, eval_supervised_metrics))
                else:
                    if args.num_cluster > 0:
                        num_cluster = args.num_cluster
                        print("cluster num is set to {}".format(num_cluster))
                        best_pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embeddings).labels_
                    else:
                        best_pd_labels = None
                        
    # save feature & labels
    np.savetxt(os.path.join(save_path, "feature_scCM_{}.csv".format(dataset_name)), embeddings, delimiter=',')

    if best_pd_labels is not None:
        pd_labels_df = pd.DataFrame(best_pd_labels, columns=['kmeans'])
        pd_labels_df.to_csv(os.path.join(save_path, "pd_label_scCM_{}.csv".format(dataset_name)))

    if train_dataset.label is not None:
        label_decoded = [train_dataset.label_decoder[i] for i in gt_labels]
        save_labels_df = pd.DataFrame(label_decoded, columns=['x'])
        save_labels_df.to_csv(os.path.join(save_path, "gt_label_scCM{}.csv".format(dataset_name)))

        if best_pd_labels is not None:
            # write metrics into txt
            best_metrics = best_eval_supervised_metrics
            txt_path = os.path.join(save_path, "metric_scCM.txt")
            f = open(txt_path, "a")
            record_string = dataset_name
            for key in best_metrics.keys():
                record_string += " {}".format(best_metrics[key])
            record_string += "\n"
            f.write(record_string)
            f.close()
    total_time = time.time() - start # Add this line to calculate the total training time
    print(f"Total training time: {total_time:.2f} seconds") # Add this line to print the total training time

def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e') 
    losses = AverageMeter('Loss', ':.4e') 
    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))    
    model.train() 
    end = time.time()
    start = end # Add this line to track the start time of training
    iters_per_epoch = len(train_loader) 
    moco_m = args.moco_m               
    for i,(images, index, label)in enumerate(train_loader):
        data_time.update(time.time() - end)         
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)  
        if args.moco_m_cos:       
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args) 
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)            
        with torch.cuda.amp.autocast(True): 
            loss = model(x1=images[0], x2=images[1])
        losses.update(loss.item(), images[0].size(0)) 
        optimizer.zero_grad() 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end) 
        end = time.time()         
    progress.display(i+1)
    unsupervised_metrics = {"loss": losses.avg}   
    total_time = time.time() - start # Add this line to calculate the total training time
    print(f"Total training time: {total_time:.2f} seconds") # Add this line to print the total training time
    return unsupervised_metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args): 
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_moco_momentum(epoch, args):  
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m

def inference(eval_loader, model):
    print('Inference...')
    model.eval()
    features = []
    labels = []
    for i, (images, index, label) in enumerate(eval_loader):
        images = images.cuda()
        with torch.no_grad():
            feat = model(images, is_eval=True) 
        feat_pred = feat.data.cpu().numpy()
        label_true = label
        features.append(feat_pred)
        labels.append(label_true)    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
if __name__ == '__main__':
    main()