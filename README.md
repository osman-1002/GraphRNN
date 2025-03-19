# GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://arxiv.org/abs/1802.08773) (ICML 2018)

## Installation
Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over PyTorch v1.13.1 with Python 3.7.16.
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Then install the other dependencies.
```bash
pip install -r requirements.txt
```

## Test run
```bash
python main.py
```

## Code description
For the GraphRNN model:
`main.py` is the main executable file, and specific arguments are set in `args.py`.
`train.py` includes training iterations and calls `model.py` and `data.py`
`create_graphs.py` is where we prepare target graph datasets.

For baseline models: 
* B-A and E-R models are implemented in `baselines/baseline_simple.py`.
* [Kronecker graph model](https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf) is implemented in the SNAP software, which can be found in `https://github.com/snap-stanford/snap/tree/master/examples/krongen` (for generating Kronecker graphs), and `https://github.com/snap-stanford/snap/tree/master/examples/kronfit` (for learning parameters for the model).
* MMSB is implemented using the EDWARD library (http://edwardlib.org/), and is located in
  `baselines`.
* the DeepGMG model was implemented based on the instructions of their [paper](https://arxiv.org/abs/1803.03324) in `main_DeepGMG.py`.
* the GraphVAE model was implemented based on the instructions of their [paper](https://arxiv.org/abs/1802.03480) in `baselines/graphvae`.


# GraphRNN Modifications Report

## 1. Introduction
This report highlights the key modifications made over the original GraphRNN, focusing on architectural improvements and their potential benefits. These changes introduce refinements aimed at enhancing the model’s ability to process sequential data while incorporating graph structures.

## 2. Key Modifications

### 2.1 Integration of Graph Embedding
- In original `GRU_plain`, the GRU processes raw input or an embedded representation.
- `GRU_plain_dec` introduces a graph embedding that is concatenated with the input sequence.
- This enhancement allows the model to incorporate global structural information, potentially improving sequence generation by leveraging prior context.

### 2.2 Data-Driven Hidden State Initialization
- `GRU_plain` initializes the hidden state as a zero tensor, offering no prior information.
- `GRU_plain_dec` instead initializes the hidden state using the graph embedding.
- This enables the model to start from a more meaningful latent state, improving learning efficiency and the handling of long-term dependencies.

### 2.3 Concatenation of Additional Features in GRU Input
- In `GRU_plain`, the GRU processes only input features.
- `GRU_plain_dec` concatenates edge sequence embeddings with the graph embedding before feeding them into the GRU.
- This modification enhances the model’s ability to capture structural dependencies within the input data.

### 2.4 Structured Weight Initialization
- `GRU_plain_dec` centralizes weight initialization in a dedicated `_init_weights()` method.

## 3. Addition of GraphEncoder
The changes include an additional component, `GraphEncoder`, which extracts graph-level embeddings. This module comprises:

- **Graph Convolutions (GCN Layer):** Processes node features to extract meaningful representations.
- **Global Mean Pooling:** Aggregates node features into a fixed-size embedding.
- **Linear Projection:** Matches the output dimension to the GRU hidden state size.

By incorporating `GraphEncoder`, `GRU_plain_dec` improves context-awareness and enhances its ability to generate sequences with structural information.

## 4. Comparison of Forward Methods

### 4.1 `GRU_plain` Forward Method
- Processes the input sequence directly through the GRU network.
- Initializes the hidden state as a zero tensor, providing no prior knowledge.
- Passes the sequence through the RNN without considering graph structures.
- A straightforward approach but lacks structural awareness.

### 4.2 `GRU_plain_dec` Forward Method
- Integrates graph embeddings into sequence processing.
- Uses a graph embedding from `GraphEncoder` instead of a zero-initialized hidden state.
- Normalizes and concatenates the edge sequence with input features before passing them into the GRU.
- This approach enables the model to leverage both sequential and graph-based dependencies.

## 5. Summary
`GRU_plain_dec` introduces several refinements to improve the handling of graph-based sequential data. The key improvements include:

- Integration of graph embeddings for better contextual understanding.
- More informed hidden state initialization for improved learning efficiency.
- Structured weight initialization to enhance consistency.
- `GraphEncoder` integration for richer graph-based information.

The aim here is to offer incremental improvements that make the model more effective in handling structured graph-related data.



After stabilizing the current version, it is also planned to apply these implemetations to other rnn models.


Parameter setting:
To adjust the hyper-parameter and input arguments to the model, modify the fields of `args.py`
accordingly.
For example, `args.cuda` controls which GPU is used to train the model, and `args.graph_type`
specifies which dataset is used to train the generative model. See the documentation in `args.py`
for more detailed descriptions of all fields.

## Outputs
There are several different types of outputs, each saved into a different directory under a path prefix. The path prefix is set at `args.dir_input`. Suppose that this field is set to `./`:
* `./graphs` contains the pickle files of training, test and generated graphs. Each contains a list
  of networkx object.
* `./eval_results` contains the evaluation of MMD scores in txt format.
* `./model_save` stores the model checkpoints
* `./nll` saves the log-likelihood for generated graphs as sequences.
* `./figures` is used to save visualizations (see Visualization of graphs section).

## Evaluation
The evaluation is done in `evaluate.py`, where user can choose which settings to evaluate.
To evaluate how close the generated graphs are to the ground truth set, we use MMD (maximum mean discrepancy) to calculate the divergence between two _sets of distributions_ related to
the ground truth and generated graphs.
Three types of distributions are chosen: degree distribution, clustering coefficient distribution.
Both of which are implemented in `eval/stats.py`, using multiprocessing python
module. One can easily extend the evaluation to compute MMD for other distribution of graphs.

the orbit counts for each graph is also computed, represented as a high-dimensional data point. the MMD
between the two _sets of sampled points_ is then computed using ORCA (see http://www.biolab.si/supp/orca/orca.html) at `eval/orca`. 
One first needs to compile ORCA by 
```bash
g++ -O2 -std=c++11 -o orca orca.cpp` 
```
in directory `eval/orca`.
(the binary file already in repo works in Ubuntu). 

To evaluate, run 
```bash
python evaluate.py
```
Arguments specific to evaluation is specified in class
`evaluate.Args_evaluate`. Note that the field `Args_evaluate.dataset_name_all` must only contain
datasets that are already trained, by setting args.graph_type to each of the datasets and running
`python main.py`.

## Visualization of graphs
The training, testing and generated graphs are saved at 'graphs/'.
One can visualize the generated graph using the function `utils.load_graph_list`, which loads the
list of graphs from the pickle file, and `util.draw_graph_list`, which plots the graph using
networkx. 


## Misc
Jesse Bettencourt and Harris Chan have made a great [slide](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) introducing GraphRNN in Prof. David Duvenaud’s seminar course [Learning Discrete Latent Structure](https://duvenaud.github.io/learn-discrete/).

