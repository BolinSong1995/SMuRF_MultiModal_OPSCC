import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='..\data', help="datasets path")
    parser.add_argument('--checkpoints_dir', type=str,
                        default='..\checkpoints', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='raw_images',
                        help='name of the project. It decides where to store samples and models')

    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--optimizer_type', type=str, default='adagrad')
    parser.add_argument('--feature_type', type=str, default='raptomic',
                        help='modalities being used. radiology/pathology/raptomic')
    parser.add_argument('--act_type', type=str,
                        default='relu', help='activation function')

    parser.add_argument('--fusion_type', type=str, default='fused_attention')
    parser.add_argument('--task', type=str, default='multitask')
    parser.add_argument('--mmo_loss', type=bool, default=False)
    parser.add_argument('--cv', type=bool, default=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--hidden_units', default=(64,16), type=tuple,
                        help='tuple of hidden layers')
    parser.add_argument('--print_freq', default=1, type=int,
                        help='frequency of model checkpoint saving')
    parser.add_argument('--lr', default=0.002, type=float,
                        help='learning rate')
    parser.add_argument('--lr_policy', default='constant',
                        type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Drop out')
    parser.add_argument('--cv_splits', default=5, type=int,
                        help='Number of splits for cross validation')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='weight of the mmo loss relative to Cox loss')
    parser.add_argument('--weight_decay', default = 0, type=float,
                        help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--niter', type=int, default=0,
                        help='# of iter at starting learning rate')
    parser.add_argument('--dim_out', type=int, default=24,
                        help='final dimension after attention module')
    parser.add_argument('--feature_size', type=int, default=24,
                        help='number of input channels for Swin Transformer')
    parser.add_argument('--epoch_count', type=int,
                        default=1, help='start of epoch')

    args = parser.parse_args()
    print_options(parser, args)
    args = parse_gpuids(args)
    return args


def print_options(parser, args):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / args.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    exp_dir = os.path.join(args.checkpoints_dir, args.exp_name)
    mkdirs(exp_dir)
    model_dir = os.path.join(exp_dir, args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)+'_'+str(args.feature_type))
    mkdirs(model_dir)
    file_name = os.path.join(model_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(args):
    # set gpu ids
    if len(args.gpu_ids) > 0:
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)

        torch.cuda.set_device(args.gpu_ids[0])

    return args


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
