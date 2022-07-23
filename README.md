# Create node embedding 

The aim of this section is to create the node embedding of SRT of [spage2vec](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7983892/)  



## Work Flow
![](https://github.com/yangfeizZZ/spage2vec/blob/main/image/pipeline_create_node_embedding.png)

## Requirements
This script is a python script,and following package are required: see [yml](https://github.com/yangfeizZZ/spage2vec/blob/main/spage2vec.yml)

## Use the script 
This script can create node embedding of spatial transcriptome.What is node embedding ,see [work Flow](https://github.com/yangfeizZZ/spage2vec/blob/main/image/pipeline_create_node_embedding.png) or see [spage2vec](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7983892/). If you want to generate the node of embedding of your own spatoal transcriptome data,you can use this script.

If you want tu use this script,you should provide two txt of your own spatial transcriptome data.

1. The spatial transcriptome of genelist. see [genelist](https://raw.githubusercontent.com/yangfeizZZ/spage2vec/main/example/genelist.csv).The Q value is the probability of the gene belong to this cell.The parentCell is the cell.
2. The spatial transcriptome of taglist. see  [taglist](https://github.com/yangfeizZZ/spage2vec/blob/main/example/taglist.csv)

If you prepare the tow matrix,you can compute the SVG list by follow command

```python
$ python creat_node_embedding.py --barcodes_df=genelist.csv --taglist_df=taglist.csv
```

The result of comand is [node_embedding.npy](https://github.com/yangfeizZZ/spark/blob/master/example/SVG.txt).The resule can be used to downstream analysis.

# Downstream

The aim of this section is to analysis the node embedding.

## Work Flow
![](https://github.com/yangfeizZZ/spage2vec/blob/main/image/pipeline_downstream_analysis.png)

## Requirement
This script is a python script,and following package are required: see [yml](https://github.com/yangfeizZZ/spage2vec/blob/main/spage2vec.yml)

## Use the script
This script can analysis the node embedding.

If you want to use this script,you should provide three files of your own spatial transcriptome data.

1. The spatial transcriptome of genelist. see [genelist](https://raw.githubusercontent.com/yangfeizZZ/spage2vec/main/example/genelist.csv).The Q value is the probability of the gene belong to this cell.The parentCell is the cell.
2. The spatial transcriptome of taglist. see  [taglist](https://github.com/yangfeizZZ/spage2vec/blob/main/example/taglist.csv)
3. The node embedding.see [node_embedding.npy](https://github.com/yangfeizZZ/spark/blob/master/example/SVG.txt)

If you prepare the three file,you can analysis by follow command

```python
$ python plot.py --barcodes_df=genelist.csv --taglist_df=taglist.csv --node_embedding=node_embedding.npy
```

This script will generate three png. 
![gene.png](https://github.com/yangfeizZZ/spage2vec/blob/main/example/gene.png) ;
![embebding_spot.png](https://github.com/yangfeizZZ/spage2vec/blob/main/example/embebding_right_spot.png) ;
![jubu.png](https://github.com/yangfeizZZ/spage2vec/blob/main/example/jubu.png)
![fangda.png](https://github.com/yangfeizZZ/spage2vec/blob/main/example/fangda.png)