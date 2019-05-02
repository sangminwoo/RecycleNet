import argparse
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from data import TrashDataset
import ResNet 
from utils import indexes_to_one_hot
import matplotlib.pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser(description="TrashNet")
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--gpu", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--resume1', type=str, default=None)
    parser.add_argument('--resume2', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    # parser.add_argument('--lr_finetune', type=float, default=5e-5)
    # parser.add_argument('--save_model_interval', type=int, default=5000)
    # parser.add_argument('--save_training_img_interval', type=int, default=5000)
    # parser.add_argument('--vis_interval', type=int, default=5)
    # parser.add_argument('--max_iter', type=int, default=1000000)
    # parser.add_argument('--display_id', type=int, default=10)
    parser.add_argument('--att_mode', type=str, default='cbam', help='attention module mode: ours, cbam, se')
    parser.add_argument('--use_att', action='store_true', help='use attention module')
    parser.add_argument('--no_pretrain', action='store_true', help='training from scratch')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--adjust-freq', type=int, default=20, help='learning rate adjustment frequency (default: 20)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
    return parser.parse_args()


def example_extract(output1, output2, target, input_paths, file, topk=(1,)):
    # with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred1 = output1.topk(maxk, 1, True, True)
    _, pred2 = output2.topk(maxk, 1, True, True)

    pred1 = pred1.t()
    pred2 = pred2.t()

    correct1 = pred1.eq(target.view(1, -1).expand_as(pred1))
    correct2 = pred2.eq(target.view(1, -1).expand_as(pred2))
    mask = (correct1==1) & (correct2==0)
    mask = mask.squeeze()
    indices = (mask==1).nonzero()

    # better_examples = []
    if indices.size(0) > 0:
        for idx in indices:
            file.write(input_paths[idx])
            file.write('\n')

    # return better_examples




def test(args, val_loader, model1, model2, device, file):
        model1.eval()
        model2.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target, input_path) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.to(device)

                target = torch.from_numpy(np.asarray(target))
                target = target.to(device)

                output1 = model1(input)
                output2 = model2(input)

                example_extract(output1, output2, target, input_path, file)

if __name__ == '__main__':
    args = get_arguments()

    BATCH_SIZE = args.b
    GPU = args.gpu
    ROOT_DIR = args.root_dir
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    if not args.evaluate:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    def ToCudaVariable(xs, volatile=False, requires_grad=True):
        if torch.cuda.is_available():
            return [Variable(x.cuda(), volatile=volatile, requires_grad=requires_grad) for x in xs]
        else:
            return [Variable(x, volatile=volatile, requires_grad=requires_grad) for x in xs]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model1 = nn.DataParallel( resnet.resnet18(pretrained=False, num_classes=6, use_att=True, att_mode=args.att_mode).to(device) )

    model2 = nn.DataParallel( resnet.resnet18(pretrained=False, num_classes=6, use_att=False, att_mode=args.att_mode).to(device) )


    if args.resume1 and args.resume2:
        if os.path.isfile(args.resume1) and os.path.isfile(args.resume2):
            print("=> loading checkpoint '{}'".format(args.resume1))
            checkpoint = torch.load(args.resume1)
            best_acc1 = checkpoint['best_acc1']
            model1.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume1, checkpoint['epoch']))
            print('=> best accuracy {}'.format(best_acc1))

            print("=> loading checkpoint '{}'".format(args.resume2))
            checkpoint = torch.load(args.resume2)
            best_acc1 = checkpoint['best_acc1']
            model2.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume2, checkpoint['epoch']))
            print('=> best accuracy {}'.format(best_acc1))

    cudnn.benchmark = True

    test_img_transform1 = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
    test_img_transform2 = transforms.Normalize(mean=MEAN, std=STD)
    # test_dataset = TrashDataset(ROOT_DIR, test_img_transform, 'test')
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=True)


    # f = open("better_examples.txt", "w")
    # test(args, test_loader, model1, model2, device, f)

    with torch.no_grad():
        f = open("better_examples.txt", "r")
        with f as lines:
            ori_imgs = []
            better_imgs = []
            for line in lines:
                path = line.rstrip('\n')
                from PIL import Image
                ori_img = test_img_transform1( Image.open( path ).convert('RGB') )
                ori_imgs.append(ori_img)
                img = test_img_transform2(ori_img)
                better_imgs.append(img)

            better_imgs = torch.stack(better_imgs)

            _, att = model1(better_imgs)
            _, base = model2(better_imgs)
            
            att = att.mean(1)
            base = base.mean(1)
            att_norms = []
            base_norms = []
            for b_ in range(att.size(0)):
                att_norm = (att[b_] - torch.min(att[b_])) / (torch.max(att[b_]) - torch.min(att[b_]))
                base_norm = (base[b_] - torch.min(base[b_])) / (torch.max(base[b_]) - torch.min(base[b_]))
                att_norms.append(att_norm)
                base_norms.append(base_norm)
            att_norms = torch.stack(att_norms)
            base_norms = torch.stack(base_norms)
            ori_imgs = torch.stack(ori_imgs)

            row = 3
            col = att_norms.size(0)

            # import pdb
            # pdb.set_trace()

            plt.figure(1)
            # plt.figure(2)
            k = 1
            for c in range(col):
                if c in [2,3,4,6,7]:
                    plt.subplot(1,3,1); plt.imshow(att_norms[c])
                    plt.subplot(1,3,2); plt.imshow(base_norms[c])
                    plt.subplot(1,3,3); plt.imshow(ori_imgs[c].permute(1,2,0))
                    plt.savefig('better_{}.png'.format(k))
                    k+=1
                    plt.gcf().clear()
                    plt.subplot(row, col, k); plt.imshow(att_norms[c])
                    plt.subplot(row, col, col+k); plt.imshow(base_norms[c])
                    plt.subplot(row, col, 2*col+k); plt.imshow(ori_imgs[c].permute(1,2,0))

            plt.show()
            # plt.figure(1)
            # import pdb
            # pdb.set_trace()
