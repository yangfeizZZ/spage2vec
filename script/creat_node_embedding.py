__author__= 'yangfei'
__date__  = '2022.6.7'

import os
import datetime
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

from tensorflow.keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from tensorflow.keras.utils import to_categorical
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.layer.graphsage import AttentionalAggregator
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from stellargraph import globalvar
from numpy.random import seed
from tensorflow import set_random_seed

seed(42)
set_random_seed(42)

parser=argparse.ArgumentParser()
parser.add_argument('-bd',"--barcodes_df",required=True)
parser.add_argument("-td","--taglist_df",required=True)
args =  parser.parse_args()


# Auxiliary function to compute d_max
def plotNeighbor(barcodes_df):
    barcodes_df.reset_index(drop=True, inplace=True)

    kdT = KDTree(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T)
    d,i = kdT.query(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T,k=2)
    plt.hist(d[:,1],bins=200);
    plt.axvline(x=np.percentile(d[:,1],97),c='r')
    print(np.percentile(d[:,1],97))
    d_th = np.percentile(d[:,1],97)
    return d_th

# Load gene panel taglist
tagList_df = pd.read_csv(args.tagList_df)
#tagList_df =pd.DataFrame(['3110035E14Rik','6330403K07Rik','Adgrl2','Aldoc','Arpp21','Atp1b1','Bcl11b','Cadps2','Calb1','Calb2','Calm2','Cck','Cdh13','Chodl','Chrm2','Cnr1','Col25a1','Cort','Cox6a2','Cplx2','Cpne5','Crh','Crhbp','Cryab','Crym','Cux2','Cxcl14','Enc1','Enpp2','Fam19a1','Fos','Fxyd6','Gabrd','Gad1','Gap43','Gda','Grin3a','Hapln1','Htr3a','Id2','Kcnk2','Kctd12','Kit','Lamp5','Lhx6','Ndnf','Neurod6','Nos1','Nov','Npy','Npy2r','Nr4a2','Nrn1','Nrsn1','Ntng1','Pax6','Pcp4','Pde1a','Penk','Plcxd2','Plp1','Pnoc','Prkca','Pthlh','Pvalb','Pvrl3','Qrfpr','Rab3c','Rasgrf2','Rbp4','Reln','Rgs10','Rgs12','Rgs4','Rorb','Rprm','Satb1','Scg2','Sema3c','Serpini1','Slc17a8','Slc24a2','Slc6a1','Snca','Sncg','Sst','Sulf2','Synpr','Tac1','Tac2','Th','Thsd7a','Tmsb10','Trp53i11','Vip','Vsnl1','Wfs1','Yjefn3','Zcchc12'], columns=['Gene'])
# Load spot data 
barcodes_df = pd.read_csv(args.barcodes_df, sep = ",", names=['Gene', 'global_Y_pos', 'global_X_pos', 'Q', 'parentCell'],header=0)
# Compute d_max for generating spatial graph
d_th = plotNeighbor(barcodes_df)

# Auxiliary function to build spatial gene expression graph
def buildGraph(barcodes_df, d_th, tagList_df):
    G = nx.Graph()
    features =[]
    barcodes_df.reset_index(drop=True, inplace=True)

    gene_list = tagList_df.Gene.values
    # Generate node categorical features
    one_hot_encoding = dict(zip(gene_list,to_categorical(np.arange(gene_list.shape[0]),num_classes=gene_list.shape[0]).tolist()))
    barcodes_df["feature"] = barcodes_df['Gene'].map(one_hot_encoding).tolist()
    features.append(np.vstack(barcodes_df.feature.values))

    kdT = KDTree(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T)
    res = kdT.query_pairs(d_th)
    res = [(x[0],x[1]) for x in list(res)]

    # Add nodes to graph
    G.add_nodes_from((barcodes_df.index.values), test=False, val=False, label=0)
    # Add node features to graph
    nx.set_node_attributes(G,dict(zip((barcodes_df.index.values), barcodes_df.feature)), 'feature')
    # Add edges to graph
    G.add_edges_from(res)

    return G, barcodes_df

# Build spatial gene expression graph
G, barcodes_df = buildGraph(barcodes_df, d_th, tagList_df)

# Remove components with less than N nodes
N=3
for component in tqdm(list(nx.connected_components(G))):
    if len(component)<N:
        for node in component:
            G.remove_node(node)

# Remove spots without cell-type label from Qian et al.
barcodes_df = barcodes_df[(barcodes_df.parentCell!=0)]

G = sg.StellarGraph(G, node_features="feature")
nodes = list(G.nodes())
number_of_walks = 1
length = 2
unsupervised_samples = UnsupervisedSampler(G, nodes=nodes, length=length, number_of_walks=number_of_walks, seed=42)
batch_size = 50
epochs = 50
num_samples = [20, 10]
train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples, seed=42).flow(unsupervised_samples)

layer_sizes = [50, 50]
assert len(layer_sizes) == len(num_samples)

graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=train_gen, aggregator=AttentionalAggregator, bias=True, dropout=0.0, normalize="l2", 
        kerne_regularizer='l1')
# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.build()
prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method='ip')(x_out)

logdir = os.path.join("logs", datetime.datetime.now().strftime("pciSeq-%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.5e-4),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)
#训练神经网络模型
history = model.fit_generator(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=True,
    workers=12,
    shuffle=True,
    callbacks=[tensorboard_callback,earlystop_callback]
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

# Save the model
embedding_model.save('./nn_model.h5')
embedding_model.compile(
    optimizer=keras.optimizers.Adam(lr=0.5e-4),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)

nodes = list(G.nodes())
node_gen = GraphSAGENodeGenerator(G, 50, [20,10], seed=42).flow(nodes)
node_embeddings = embedding_model.predict_generator(node_gen, workers=12, verbose=1)
np.save('./embedding_ISS_right.npy',node_embeddings)
