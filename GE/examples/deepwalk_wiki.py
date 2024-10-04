# %%
# gensim==3.6.0'
import numpy as np


from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import sys
sys.path.append('../')
from packages.classify import read_node_label, Classifier
from packages import DeepWalk
from packages.cosine_similar import cosine_similar
# %%
if __name__ == "__main__":
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    # 尋找餘弦相似
    cs = cosine_similar(embeddings)
    cs.most_similar(['1397'])
# %%
