import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from collections import OrderedDict

r50=torch.load('weights/ssd_300_VOC0712_0.pth')
rr50=OrderedDict()

#old_key=
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
    
torch.save(r50, 'weights/ssd_300_VOC0712.pth')    


#vgg=torch.load('vgg16_reducedfc.pth')

'''for i,_ in enumerate(range(len(vgg))):
    k, v = vgg.popitem(False)
    print(i,k)'''