from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd

from layers.box_utils import jaccard
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model', default='vgg',
                    help='model architecture of the base network')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    mAP=0;
    for i in range(num_images):
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        
        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
           
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        
        detections2=detections[0,:,:,0].view(-1)
        detectionr, rind =detections2.sort(descending=True)
        ntop=40
        jj=rind[:ntop]%200
        ii=rind[:ntop]/200
        
        gt_label=[box[-1]  for box in annotation]
        possible=len(gt_label) 
        if gt_label==0:
            continue
        gt_box=torch.tensor([box[:-1]  for box in annotation])
        dt_box=detections[0,ii,jj,1:]*scale
        iou=jaccard(dt_box,gt_box)
        k=0; correct=0;  
        precisions=[];
        while correct<possible and k<ntop:
            flag=False
            for j, gtl in enumerate(gt_label):
                if ii[k]-1==gtl and iou[k,j]>0.5:
                    correct+=1
                    flag=True
                    gt_label[j]=0 #turn the ground truth off
                    break
            if flag:
                name = labelmap[ii[k]-1]
                conf = detectionr[k].item()*100
                precision=correct/(k+1);recall=correct/possible
                precisions.append(precision)                    
                gtb=gt_box[j,:];dtb=dt_box[k,:]; iou1=iou[k,j].item()
                with open(filename, mode='a') as f:
                    f.write('rank %d: %s(%d), score %1.2f%%, precision:%1.2f, recall:%1.2f\n'%
                      (k,name,ii[k]-1,conf,precision,recall))
                    f.write('detection:'+' ||'.join('%d'%c.item() for c in dtb)+
                      ', ground truth: '+' ||'.join('%d'%c.item() for c in gtb)+', iou:%1.2f\n'%iou1)
                
            k+=1
        AP=1;
        for j,p in enumerate(precisions):
            prec=np.max(np.array(precisions[j:]))
            AP+=prec*(np.floor(np.float(j+1)/possible/0.1)-np.floor(np.float(j)/possible/0.1))
        AP=AP/11*100
        mAP=(mAP*i+AP)/(i+1)
        with open(filename, mode='a') as f:
            f.write('---AP: %.1f%%, total AP: %.1f%%---'%(AP,mAP))
        print('Image {:d}/{:d} AP: {:.1f}, total AP: {:.1f}'.format(i+1, num_images,AP,mAP))
        

def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test',args.model, 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
