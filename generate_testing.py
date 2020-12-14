# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:44:12 2020

@author: gxjco
"""
import networkx as nx
import numpy as np
#erdos-renyi graph for testing
#every 25 pair is a batch, with one factor fixed and other factor varies
graphs_a1=[]
graphs_a2=[]
graphs_b1=[]
graphs_b2=[]
graphs_c1=[]
graphs_c2=[]
graphs_a1_nodes=[]
graphs_a2_nodes=[]
graphs_b1_nodes=[]
graphs_b2_nodes=[]
graphs_c1_nodes=[]
graphs_c2_nodes=[]
num_node=20
for batch in range(500):
    a=np.random.randint(2,12)
    for i in range(25):
        b=np.random.rand(1)[0]
        c=np.random.randint(0,10)
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_a1.append(nx.to_numpy_array(g))
        graphs_a1_nodes.append(node)
        
        b=np.random.rand(1)[0]
        c=np.random.randint(0,10)
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_a2.append(nx.to_numpy_array(g))
        graphs_a2_nodes.append(node)
        
    b=np.random.rand(1)[0]
    for i in range(25):
        a=np.random.randint(2,12)
        c=np.random.randint(0,10)
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_b1.append(nx.to_numpy_array(g))
        graphs_b1_nodes.append(node)

        a=np.random.randint(2,12)
        c=np.random.randint(0,10)        
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_b2.append(nx.to_numpy_array(g))
        graphs_b2_nodes.append(node)
        
    c=np.random.randint(0,10)
    for i in range(25):
        a=np.random.randint(2,12)
        b=np.random.rand(1)[0]
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_c1.append(nx.to_numpy_array(g))
        graphs_c1_nodes.append(node)
        
        a=np.random.randint(2,12)
        b=np.random.rand(1)[0]
        node=np.zeros((num_node,2))
        g=nx.generators.random_graphs.watts_strogatz_graph(num_node,a,b)
        node[:,0]=np.random.normal((c+1)*10,1,num_node)
        node[:,1]=np.random.normal(b*100,1,num_node)       
        graphs_c2.append(nx.to_numpy_array(g))
        graphs_c2_nodes.append(node)      
        
np.save('WS_graph_testing_a1.npy',graphs_a1)
np.save('WS_graph_testing_a2.npy',graphs_a2)
np.save('WS_graph_testing_b1.npy',graphs_b1)
np.save('WS_graph_testing_b2.npy',graphs_b2)
np.save('WS_graph_testing_c1.npy',graphs_c1)
np.save('WS_graph_testing_c2.npy',graphs_c2)
np.save('WS_graph_testing_a1_nodes.npy',graphs_a1_nodes)
np.save('WS_graph_testing_a2_nodes.npy',graphs_a2_nodes)
np.save('WS_graph_testing_b1_nodes.npy',graphs_b1_nodes)
np.save('WS_graph_testing_b2_nodes.npy',graphs_b2_nodes)
np.save('WS_graph_testing_c1_nodes.npy',graphs_c1_nodes)
np.save('WS_graph_testing_c2_nodes.npy',graphs_c2_nodes)