import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from collections import OrderedDict
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

r50=model_zoo.load_url(model_urls['resnet50'])
rr50=OrderedDict()

nk0={'conv1':'0','bn1':'1'}

bb=np.array([3,4,6,3])
#old_key=
for i,_ in enumerate(range(len(r50))):
    k, v = r50.popitem(False)
    #print(i,k)
    key=k.split('.')
    nk=''
    if key[0] in nk0.keys():
        key[0]=nk0[key[0]]
        nk=".".join(key)
        r50[nk]=v
    elif key[0].startswith('layer'):
        layer=int(key[0][-1])-1
        key[0]= str(bb[:layer].sum()+int(key[1])+4)
        del key[1]
        nk=".".join(key)
        r50[nk]=v
    print(i,nk,k)    
    
torch.save(r50, 'resnet50.pth')    
