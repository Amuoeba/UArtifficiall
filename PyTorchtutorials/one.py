# -*- coding: utf-8 -*-
import torch
import numpy as np

x = torch.Tensor(5,3)
print "----- X -----"
print x

c = torch.rand(5,3)
print "----- C -----"
print c
y= torch.rand(5,3)
print "----- Y -----"
print y
v = c+y
print "----- V -----"
print v

#adding in place
print "----- ADDING IN PLACE -----"
y.add_(c)
print "Y is now",y

#adding to a placeholder
result = torch.Tensor(5,3)
torch.add(y,c,out=result)
print "Result is \n", result

print result[:,1]


#Converting to numpy arrays and vice versa

tor = torch.ones(5)
print "As torch tensor: ",tor

nump = tor.numpy()
print "As numpy array: ",nump

numo1 = np.zeros(5)
tor1 = torch.from_numpy(numo1)
print tor1

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print "From cuda",(x + y)
    
