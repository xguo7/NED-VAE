# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:01:08 2020

@author: gxjco
"""

import networkx as nx
import numpy as np
#erdos-renyi graph for testing
#this test is used for MIG, SAP metric evaluation
graphs=[]
factors=[]
graph_nodes=[]

num_node=20
for batch in range(25000):
        a=np.random.randint(2,12)
        b=np.random.rand(1)[0]
        c=np.random.randint(0,10)
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)     
        graphs.append(nx.to_numpy_array(g))
        graph_nodes.append(node)
        factors.append([a,b,c])
np.save('WS_graph_testing2.npy',graphs)
np.save('WS_node_testing2.npy',graph_nodes)
np.save('WS_factor_testing2.npy',factors)        
