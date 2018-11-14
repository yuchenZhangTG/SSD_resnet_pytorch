import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from collections import OrderedDict

if torch.cuda.is_available():
    r50=torch.load('ssd_300_VOC0712_0.pth')
else:
    r50=torch.load('ssd_300_VOC0712_0.pth',map_location='cpu')
rr50=OrderedDict()


for i,_ in enumerate(range(len(r50))):
    k, v = r50.popitem(False)
    #print(i,k)
    key=k.split('.')
    nk=''
    if key[0]=='vgg':
        key[0]='base'
        nk=".".join(key)
        r50[nk]=v
    else:
        r50[k]=v
    
torch.save(r50, 'ssd_300_VOC0712.pth')    
