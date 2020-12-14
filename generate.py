# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:02:03 2019

@author: gxjco
"""

import networkx as nx
import numpy as np



num_node=20
num_sample=50
#watts_strogatz graph for training 
          
          
node_att=[]
graphs=[]          
for a in range(2,12):
   for b in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.99]:  
        for c in range(10):
            for i in range(num_sample):
                node=np.zeros((num_node,2))
                node[:,0]=np.random.normal((c+1)*10,1,num_node)
                node[:,1]=np.random.normal(b*100,1,num_node)
                G=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
                node_att.append(node)
                graphs.append(nx.to_numpy_array(G)) 
                 
graphs=np.array(graphs).reshape(-1,num_node,num_node)
node_att=np.array(node_att).reshape(-1,num_node,2)
np.save('WS_graph_node.npy',node_att)
np.save('WS_graph.npy',graphs)





