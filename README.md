# Learning on Streaming Graphs with Experience Replay

This project applies experience replay to enable continuous graph representation learning in the streaming setting.
We update an Inductive Graph Neural Network while graph changes arrive as a stream or vertices or edges.

This repository contains the implementation of the following online training methods:
* Random-Based Rehearsal-RBR: yields a uniform sample of the entire training graph 
* Priority-Based Rehearsal-PBR: prioritizes datapoints based on the model prediction error

We also provide the following baseline implementations:
* No-rehearsal: trains over new vertices only
* Offline: performs multiple epochs over the entire graph


## Getting Started

* Clone the repository
* Install the dependencies
* This code automatically downloads the required datasets.
  * You can also download and store them in /datasets/:
  * Pubmed: https://file.perini.me/graphs/pubmed.zip
  * Bitcoin: https://file.perini.me/graphs/bitcoin.zip
  * Reddit: http://www.mediafire.com/file/qpvcabh453jzhhb/reddit.zip/file
* Run the script (using python 3): 
```
python train <args>
```
or
```
python train/__main__.py <args>
```

### Prerequisites

Install the dependencies:

```
pip3 install -r requirements.txt
```

## Datasets included
* [Pubmed](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz) - Galileo Namata, et. al. "Query-driven Active Surveying for Collective Classification." MLG. 2012.
* [Reddit](http://snap.stanford.edu/graphsage/reddit.zip) - W.L. Hamilton et. al. "Inductive Representation Learning on Large Graphs", NeurIPS 2017
* [Elliptic Bitcoin](https://www.kaggle.com/ellipticco/elliptic-data-set) - Weber et. al. , "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", Anomaly Detection in Finance Workshop, 25th SIGKDD Conference on Knowledge Discovery and Data Mining

## Parameters
### Required
* ```dataset```: dataset: 'elliptic', 'pubmed', 'reddit' or 'tagged
* ```backend```: Deep learning framework. 'tf': Tensorflow (>= 2.2) or 'pytorch': Pytorch
* ```save_result```: output path
* ```save_tsne```: tsne plots path

### Optional
GPU
* ```--cuda```: Enable GPU accelleration
* ```--gpu <id>```: Use a specific GPU (CUDA must be enabled)

Model

* ```--latent_dim N```: size of the vector obtained applying a Pooling layer
* ```--embedding_size N```: size of the vector obtained applying a GraphSAGE aggregator (Pooling + Dense layers)
* ```--depth N```: sampling depth and number of GraphSAGE layers
* ```--samples N```: maximum number of neighbours sampled: in case of a vertex with degree >= of ```N```, N neighbours will be sampled
* ```--dropout N```: dropout used during the training phase

Train behaviour
* ```--batch_size N```: batch size used during the training phase (number of vertices trained together)
* ```--batch_full N```: batch size used during the evaluation/forward phase (number of vertices evaluated together)
* ```--snapshots N```: split the temporal graph into N snapshots
* ```--batch_timestep N```: train ```N``` batches in every snapshot 
* ```--eval N```: evaluate the model every ```N``` snapshots
* ```--epochs_offline N```: trains the offline model for ```N``` epochs
* ```--train_offline N```: trains the offline model every ```N``` snapshots
* ```--priority_forward N```: update the priorities running a forward pass every ```N``` snapshots
* ```--plot_tsne N```: generate a TSNE plot every N ```N``` snapshots
* ```--delta N```: evaluate the current model over a graph ```N``` snapshot in the future

### Example
```
python  train pubmed pytorch test_eval.csv tsne_1 --eval 10 --batch_timestep 20 --epochs_offline 25 --cuda
```
Runs the code using pytorch and the reddit pubmed. Results will be stored in test_eval.csv and TSNE plots in the tsne_1 folder.
The model is evaluated every 10 snapshots and 20 batches are trained in every timestep. The offline model is trained for 25 epochs. GPU accelleration is enabled.


## Add new datasets

Add a new dynamic graph:
* Create a new folder in train/datasets
* Vertex ids must be integers (0 to <n. vertices>)

In the folder the following files should be stored:

* ```graph.adjlist```: The graph adjacency list
* ```feat_data.npy```: The vertex features: a (n. vertices, #features) matrix. Each row stores the features of a vertex. The row index is the vertex id.
* ```targets.npy```: The labels: a (n. vertices, 1) matrix. Each row stores a label. The row index is the vertex id.
* ```vertex_timestamp.json``` or ```edge_timestamp.json```: In the first case, a json map <vertex id, timestamp>, in the second one, a map <edge, timestamp> where the edge is "(first vertex, second vertex)"

Then, create a <dataset>.py file in train/dataset_utils and implement:
  
```
def load(path, snapshots=100):
```

This method should return a numpy matrix of features, a numpy matrix of targets, an instance of ```DynamicGraphVertex``` or ```DynamicGraphEdge``` and the number of classes in the dataset.
