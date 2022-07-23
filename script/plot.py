import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
import umap
import umap.umap_ as umap
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree#计算点与点之间的距离
import scanpy as sc
from sklearn.preprocessing import StandardScaler#数据标准化
import seaborn as sns#绘制热图
import scipy

parser=argparse.ArgumentParser()
parser.add_argument('-bd',"--barcodes_df",required=True)
parser.add_argument("-td","--taglist_df",required=True)
parser.add_argument("-ne","--node_embedding",required=True)
#parser.add_argument("-rf","--result_file",required=True)
args =  parser.parse_args()

# Load taglist
tagList_df = pd.read_csv(args.tagList_df)
#tagList_df =pd.DataFrame(['3110035E14Rik','6330403K07Rik','Adgrl2','Aldoc','Arpp21','Atp1b1','Bcl11b','Cadps2','Calb1','Calb2','Calm2','Cck','Cdh13','Chodl','Chrm2','Cnr1','Col25a1','Cort','Cox6a2','Cplx2','Cpne5','Crh','Crhbp','Cryab','Crym','Cux2','Cxcl14','Enc1','Enpp2','Fam19a1','Fos','Fxyd6','Gabrd','Gad1','Gap43','Gda','Grin3a','Hapln1','Htr3a','Id2','Kcnk2','Kctd12','Kit','Lamp5','Lhx6','Ndnf','Neurod6','Nos1','Nov','Npy','Npy2r','Nr4a2','Nrn1','Nrsn1','Ntng1','Pax6','Pcp4','Pde1a','Penk','Plcxd2','Plp1','Pnoc','Prkca','Pthlh','Pvalb','Pvrl3','Qrfpr','Rab3c','Rasgrf2','Rbp4','Reln','Rgs10','Rgs12','Rgs4','Rorb','Rprm','Satb1','Scg2','Sema3c','Serpini1','Slc17a8','Slc24a2','Slc6a1','Snca','Sncg','Sst','Sulf2','Synpr','Tac1','Tac2','Th','Thsd7a','Tmsb10','Trp53i11','Vip','Vsnl1','Wfs1','Yjefn3','Zcchc12'], columns=['Gene'])
tagList_df.shape
# Load spot coordinates
barcodes_df = pd.read_csv(args.barcodes_df, sep = ",", names=['Gene', 'global_Y_pos', 'global_X_pos', 'Q', 'parentCell'],header=0)

#------------------Auxiliary function to build spatial gene expression graph--------------------#
def buildGraph(barcodes_df, d_th):
    G = nx.Graph()
    node_removed = []
    barcodes_df.reset_index(drop=True, inplace=True)

    kdT = KDTree(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T)
    res = kdT.query_pairs(d_th)
    res = [(x[0],x[1]) for x in list(res)]

    # Add nodes
    G.add_nodes_from((barcodes_df.index.values), test=False, val=False, label=0)
    # Add edges
    G.add_edges_from(res)

    # Remove connected components with less than N nodes, representing spurious gene expression
    N=3
    for component in tqdm(list(nx.connected_components(G))):
        if len(component)<N:
            for node in component:
                node_removed.append(node)
                G.remove_node(node)

    barcodes_df = barcodes_df[~barcodes_df.index.isin(node_removed)]
    barcodes_df.reset_index(drop=True, inplace=True)

    return G, barcodes_df

#--------------Auxiliary function to compute maximun nearst neighbor distance d_th------------#
def plotNeighbor(barcodes_df):
    barcodes_df.reset_index(drop=True, inplace=True)

    kdT = KDTree(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T)
    d,i = kdT.query(np.array([barcodes_df.global_X_pos.values,barcodes_df.global_Y_pos.values]).T,k=2)
    plt.hist(d[:,1],bins=200);
    plt.axvline(x=np.percentile(d[:,1],97),c='r')
    d_th = np.percentile(d[:,1],97)

    return d_th

d_th = plotNeighbor(barcodes_df)
G, barcodes_df = buildGraph(barcodes_df, d_th)
print("The number of nodes is "+str(G.number_of_nodes()))

#---------------------------------------Visualization of node embeddings--------------------------------------------#
X = np.load(args.node_embedding)

reducer = umap.UMAP(
    n_neighbors=10,#确定使用的相邻点的数量
    n_components=3,# default 2, The dimension of the space to embed into
    n_epochs=500,#default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings
    init='spectral',# default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}
    min_dist=0.1,## default 0.1, The effective minimum distance between embedded points
    spread=1,# default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are
    random_state=42## default: None, If int, random_state is the seed used by the random number generator;
)
embedding = reducer.fit_transform(X)

Y_umap = embedding
Y_umap -= np.min(Y_umap, axis=0)
Y_umap /= np.max(Y_umap, axis=0)
c_umap = Y_umap

fig=plt.figure(figsize=(7,4),dpi=500)
cycled = [0,1,2,0]
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=Y_umap,  s=5, marker='.', linewidths=0, edgecolors=None)
    plt.xlabel("Y"+str(cycled[i]))
    plt.ylabel("Y"+str(cycled[i+1]))
plt.tight_layout()
plt.savefig('./embebding_right_spot.png', dpi=500, bbox_inches='tight', pad_inches=0.0)

fig=plt.figure(figsize=(7,7),dpi=500)
plt.scatter(barcodes_df.global_X_pos, barcodes_df.global_Y_pos, c=Y_umap, s=2,marker='.',linewidths=0, edgecolors=None)
plt.axis('scaled');
plt.xticks([])
plt.yticks([])
plt.savefig('./gene.png', dpi=500, bbox_inches='tight', pad_inches=0.0)

import scanpy as sc
adata = sc.AnnData(X=X)
# Compute the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15)

# Run Leiden clustering algorithm
sc.tl.leiden(adata, random_state = 42)

adata.obs['Gene'] = barcodes_df.Gene.values

# Extract leiden clusters
col_cluster = adata.obs['leiden'].values.astype(int)
col = np.unique(col_cluster)

# Add leiden clusters to spots dataframe
barcodes_df['cluster'] = adata.obs['leiden'].values
go_clusters = np.unique(adata.obs['leiden'].values)

#-------------------------------------Merge cluster based on correlation distance--------------------------------#
def post_merge(df, labels, post_merge_cutoff, linkage_method='single', 
               linkage_metric='correlation', fcluster_criterion='distance', name='', save=False):
    Z = scipy.cluster.hierarchy.linkage(df.T, method=linkage_method, metric=linkage_metric)
    merged_labels_short = scipy.cluster.hierarchy.fcluster(Z, post_merge_cutoff, criterion=fcluster_criterion)

    #Update labels  
    label_conversion = dict(zip(df.columns, merged_labels_short))
    label_conversion_r = dict(zip(merged_labels_short, df.columns))
    new_labels = [label_conversion[i] for i in labels] 

    #Plot the dendrogram to visualize the merging
    fig, ax = plt.subplots(figsize=(20,10))
    scipy.cluster.hierarchy.dendrogram(Z, labels=df.columns ,color_threshold=post_merge_cutoff)
    ax.hlines(post_merge_cutoff, 0, ax.get_xlim()[1])
    ax.set_title('Merged clusters')
    ax.set_ylabel(linkage_metric, fontsize=20)
    ax.set_xlabel('pre-merge cluster labels', fontsize=20)
    ax.tick_params(labelsize=10)
    
    return new_labels

# Create cluster gene expression matrix
hm = barcodes_df.groupby(['Gene','cluster']).size().unstack(fill_value=0)

# Add Vsnl1 gene with zero expression
hm = hm.append(pd.DataFrame(np.zeros((tagList_df[~tagList_df.Gene.isin(hm.index.values)].values.reshape(-1).shape[0],hm.shape[1])), 
    index=tagList_df[~tagList_df.Gene.isin(hm.index.values)].values.reshape(-1), columns=hm.columns)).sort_index()

# Z-score normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
hm = pd.DataFrame(scaler.fit_transform(hm.values), columns=hm.columns, index=hm.index)
# Merge high correlated clusters
import scipy
hm_merge = post_merge(hm, hm.columns, 0.05, linkage_metric='correlation', linkage_method='average')

#因为hm的表达矩阵被处理成hm_merge的表达矩阵，因此需要重新创建hm表达矩阵
hm = barcodes_df.groupby(['Gene','cluster']).size().unstack(fill_value=0)

# 将taglist里有而hm里没有的基因添加到hm里并且该基因在每个群里的表达量为0
hm = hm.append(pd.DataFrame(np.zeros((tagList_df[~tagList_df.Gene.isin(hm.index.values)].values.reshape(-1).shape[0],hm.shape[1])), \
        index=tagList_df[~tagList_df.Gene.isin(hm.index.values)].values.reshape(-1), columns=hm.columns)).sort_index()

# Compute new cluster gene expression matrix based on macro clusters from merging results
hm_macro = pd.DataFrame(np.zeros((hm.shape[0], np.unique(hm_merge).shape[0])), index=hm.index, columns=np.unique(hm_merge))

for d in np.unique(hm_merge):
    hm_macro.loc[:,d] = hm.iloc[:,np.where(np.array(hm_merge)==d)[0]].sum(axis=1)

# Z-score normalization

scaler = StandardScaler()
hm_macro = pd.DataFrame(scaler.fit_transform(hm_macro.values), columns=hm_macro.columns, index=hm_macro.index)

# Annotate spots with macro cluster labels
hm_merge = np.array(hm_merge)
for macro_go in np.unique(hm_merge):
    barcodes_df.loc[barcodes_df.cluster.astype(int).isin(np.where(np.isin(hm_merge,[macro_go]))[0]),'macro_cluster'] = macro_go
# Remove spots without cell-type annotation (filter spots assgined to background 3135 and spots not assigned 0)
barcodes_assigned = barcodes_df[(barcodes_df.parentCell!=0) & (barcodes_df.parentCell!=3135)].copy()
macro_cluster_len = len(barcodes_assigned.macro_cluster.value_counts())
print(macro_cluster_len)

import random

def random_color(cluster_num):
    i = 1
    color_list= []
    while i <= cluster_num:
        i = i+1
        code_list = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        first_code = random.choice(code_list)
        second_code = random.choice(code_list)
        third_code = random.choice(code_list)
        forth_code = random.choice(code_list)
        fiveth_code = random.choice(code_list)
        sixth_code = random.choice(code_list)
        A = "#"+first_code+second_code+third_code+forth_code+fiveth_code+sixth_code
        color_list.append(A)
    return color_list
color_list = random_color(macro_cluster_len)

def cluster_num(cluster_num):
    cluster_list = []
    for i in range(1,cluster_num+1):
        cluster_list.append(i)
    return cluster_list

cluster_list = cluster_num(macro_cluster_len)

d = dict(zip(cluster_list,color_list))

from matplotlib.patches import Rectangle
# Plot Cell-type map from pciSeq
fig=plt.figure(figsize=(4,4),dpi=500)
px1 = 24500
px2 = 25500
py1 = 15000
py2 = 16000

ax = fig.add_subplot(1, 1, 1)

plt.scatter(barcodes_assigned.global_X_pos, barcodes_assigned.global_Y_pos,
            c=[d[ct] for ct in barcodes_assigned.macro_cluster],
            s=1,marker='.',linewidths=0, edgecolors=None)
roi = Rectangle((px1,py1),1000,1000,linewidth=1,edgecolor='black',facecolor='none')
plt.xticks([])
plt.yticks([])
ax.add_patch(roi);
plt.axis('scaled');
plt.savefig('./jubu.png', dpi=500, bbox_inches='tight', pad_inches=0.0)

fig=plt.figure(figsize=(4,4),dpi=500)

cut_out_df = barcodes_assigned[(barcodes_assigned.global_X_pos>px1) & (barcodes_assigned.global_X_pos<px2) & (barcodes_assigned.global_Y_pos>py1) & (barcodes_assigned.global_Y_pos<py2)]
plt.scatter(cut_out_df.global_X_pos, cut_out_df.global_Y_pos,
            c=[d[ct] for ct in cut_out_df.macro_cluster], s=10,marker='.',linewidths=0, edgecolors=None)
plt.xticks([])
plt.yticks([]);
plt.axis('scaled');
plt.savefig('./fangda.png', dpi=500, bbox_inches='tight', pad_inches=0.0)