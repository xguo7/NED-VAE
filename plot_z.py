# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:14:24 2020

@author: gxjco
This code is used to find which variabe is realted to each factor
"""

import numpy as np
import matplotlib.pyplot as plt

#edge realted factpors
data=np.load('beta-VAE_z_e.npy').reshape([25000,-1])*10e3
   
fig = plt.figure(figsize=(15,10))
n=1
for x_n in range(4):
   for y_n in range(x_n+1,4):    
    ax0 = fig.add_subplot(2,3,n)
    ax0.set_title('Scatter Plot')
    plt.xlabel('z'+str(x_n))
    plt.ylabel('z'+str(y_n))
    color=['r','b','g','c','k','y','m','coral','gold','grey']
    for a in range(10):
                ax0.scatter(data[a*2500:a*2500+1000,x_n],data[a*2500:a*2500+1000,y_n],c = color[int(a/2)],marker = 'o')   #finding the z related to a (edge related)        
    n=n+1
plt.show()


#graph realted factors
data=np.load('beta-VAE_z_n.npy').reshape([25000,-1])
   
fig = plt.figure(figsize=(15,10))
n=1
for x_n in range(4):
   for y_n in range(x_n+1,4):    
    ax0 = fig.add_subplot(2,3,n)
    ax0.set_title('Scatter Plot')
    plt.xlabel('z'+str(x_n))
    plt.ylabel('z'+str(y_n))
    color=['r','b','g','c','k','y','m','coral','gold','grey']        
    for a in range(10):
        for b in range(10):              
                ax0.scatter(data[a*2500+b*250:a*2500+b*250+100,x_n],data[a*2500+b*250:a*2500+b*250+100,y_n],c = color[int(b/2)],marker = 'o')   #finding the z related to b
    n=n+1
plt.show() 


#graph realted factors
data=np.load('beta-VAE_z_g.npy').reshape([25000,-1])
fig = plt.figure(figsize=(15,10))
n=1
for x_n in range(4):
   for y_n in range(x_n+1,4):    
    ax0 = fig.add_subplot(2,3,n)
    ax0.set_title('Scatter Plot')
    plt.xlabel('z'+str(x_n))
    plt.ylabel('z'+str(y_n))
    color=['r','b','g','c','k','y','m','coral','gold','grey']        
    for a in range(10):
        for b in range(10):  
            for c in range(10):            
                ax0.scatter(data[a*2500+b*250+c*25:a*2500+b*250+c*25+10,x_n],data[a*2500+b*250+c*25:a*2500+b*250+c*25+10,y_n],c=color[int(c/2)],marker = 'o')   #finding the z related to c
    n=n+1
plt.show() 