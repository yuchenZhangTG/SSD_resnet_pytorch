import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import torchvision
import os


#extra layers
extras = {
    'vgg': [1024,256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'resnet': [512,256,128,128,128],
}

#where to extract features
extract = {
    'vgg': {'b':[21,33],'e':[1,3,5,7]}, #vgg -14
    'resnet': {'b':[16,19],'e':[0,1,2,3]}
}


mbox = {
    'vgg': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'resnet': [4, 6, 6, 6, 4, 4],
}

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase,model, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), requires_grad=True)
        self.size = size
        self.model=model
        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        
        flag=0
        for k, v in enumerate(self.base):
            x = v(x)
            if (k-1,k)[self.model=='resnet']  in extract[self.model]['b']:
                if flag==0:
                    sources.append(self.L2Norm(x))
                    flag+=1
                else:
                    sources.append(x)
            
        '''
        for k,i in enumerate(sources[self.model]['b']):
            layer=1
            #apply base network up to source points
            for j in range(layer, (i+2,i+1)[self.model=='resnet'] ):
                x = self.base[j](x)
                layer+=1
            if k==0:
                sources.append(self.L2Norm(x))
            else:
                sources.append(x)
        '''    
            
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            if self.model=='vgg':
                x = F.relu(v(x), inplace=True)
            elif self.model =='resnet':
                x = v(x)
            if k in extract[self.model]['e']:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


'''The functions are derived from torchvision VGG and resNet
 https://github.com/pytorch/vision/blob/master/torchvision/models/'''
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def make_layers(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    bbl=block(inplanes, planes, stride, downsample)
    bbl.out_channels=planes*block.expansion
    layers.append(bbl)
    
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        bbl=block(inplanes, planes)
        bbl.out_channels=planes*block.expansion
        layers.append(bbl)
    return layers

def resnet(cfg,in_channel=3):
    layers = []
    layers += [nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    block= torchvision.models.resnet.Bottleneck
    layers += make_layers(block,64,64, cfg[0])
    layers += make_layers(block,64*block.expansion,128,cfg[1], stride=2)
    layers += make_layers(block,128*block.expansion,256,cfg[2], stride=2)
    layers += make_layers(block,256*block.expansion,512,cfg[3], stride=2)
    layers += [nn.AvgPool2d(7, stride=1)]
    return layers



def vgg_extras(cfg, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    flag = False
    in_channels=0
    for k, v in enumerate(cfg):
        if in_channels != 'S' and k>0:
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def resnet_extras(cfg):        
    layers = []
    block= torchvision.models.resnet.Bottleneck
    in_channels=0
    for k, v in enumerate(cfg):
        if k>0:
            layers += make_layers(block,in_channels,v,1,stride=(1,2)[k<3])    
        in_channels = v*block.expansion
    return layers


def multibox(base, extras, extract,cfg, num_classes):
    loc_layers = []
    conf_layers = []
    k=0
    for v in extract['b']:
        loc_layers += [nn.Conv2d(base[v].out_channels,
                        cfg[k], kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(base[v].out_channels,
                        cfg[k]*num_classes, kernel_size=3, padding=1)]
        k+=1
    for v in extract['e']:
        loc_layers += [nn.Conv2d(extras[v].out_channels,
                        cfg[k]* 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extras[v].out_channels,
                        cfg[k]*num_classes, kernel_size=3, padding=1)]
        k+=1
    return loc_layers, conf_layers


base = {
    'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    'resnet': [3, 4, 6, 3],
}


def build_ssd(phase, model, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    
    
    if model in ['vgg','resnet']:
        base_=globals()[model](base[model])
        extras_=globals()[model+'_extras'](extras[model])
        head_ = multibox(base_,extras_,extract[model], mbox[model], num_classes)    
    return SSD(phase,model, size, base_, extras_, head_, num_classes)
