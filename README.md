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

# What has been added?

Although the `GRU_plain_with_attention` is yet to produce desired results, and is with some apparent problems in its output, these are the intended changes and how it differs from the original:

##Comparison of `GRU_plain` and `GRU_plain_with_attention`


## 1. **Attention Mechanism**
- **in `GRU_plain_with_attention`**:
  - Uses a custom `Attention` class to compute attention scores over the RNN outputs.
  - Produces a context vector through a weighted sum of the RNN outputs.

---

## 2. **Layer Normalization**
- **GRU_plain_with_attention**:
  - Applies `LayerNorm` to the RNN outputs before proceeding further.

---


## 3. **Output Handling**
- **GRU_plain**:
  - Outputs the raw RNN output (or applies an output layer if `has_output=True`).
- **GRU_plain_with_attention**:
  - Uses the attention-derived `context_vector` as the primary output.
  - Optionally passes the `context_vector` through an output layer if `has_output=True`.

---

## 4. **Initialization of Parameters**
- Both classes initialize weights using Xavier initialization and biases with constant values.
- The code for parameter initialization is almost identical.

---

## 5. **Changes in `forward` Method**
| **Aspect**                 | **GRU_plain**                                | **GRU_plain_with_attention**               |
|----------------------------|---------------------------------------------|-------------------------------------------|
| **RNN Output**             | Directly uses the RNN output.               | Passes the RNN output through attention. |
| **Normalization**          | Not applied.                                | Applies layer normalization.             |
| **Attention Mechanism**    | Not present.                                | Computes attention scores and context vector. |
| **Packed Sequences**       | Handles packed sequences.                   | Handles packed sequences similarly, with additional attention logic. |
| **Output**                 | RNN output (or processed output).           | Attention-derived context vector (or processed output). |

---

## Differences Between `train_rnn_epoch_with_attention` and `train_rnn_epoch`


---

### 1. **Hidden State Handling**
   - **With Attention (`train_rnn_epoch_with_attention`)**:
     - The RNN's hidden state is manipulated by adding `hidden_null` for remaining layers after the initial hidden state.
     - The model utilizes attention to compute context vectors and attention scores (`context_vector, attention_scores = rnn(x, pack=True, input_len=y_len)`).
   
   - **Without Attention (`train_rnn_epoch`)**:
     - Hidden state handling is more straightforward, with the RNN output directly used as the hidden state for the output module.
     - No explicit attention mechanism is applied.

---

### 2. **Forward Pass**
   - **With Attention (`train_rnn_epoch_with_attention`)**:
     - The model computes `context_vector` and `attention_scores` from the RNN output. 
     - These are used as part of the hidden state in the output module: `output.hidden = torch.cat((context_vector, hidden_null), dim=0)`.
   
   - **Without Attention (`train_rnn_epoch`)**:
     - The model directly computes the hidden state for the output module without any context vector or attention-based transformations.

---

### 3. **Graph Generation**
   - **With Attention (`test_rnn_with_attention_epoch`)**:
     - During testing, the attention mechanism influences the graph prediction by using the RNN's context vector and attention scores to guide the graph generation. Attention is involved in producing outputs over multiple time steps.
   
   - **Without Attention (`test_rnn_epoch`)**:
     - The process of graph generation during testing is simpler and does not involve attention scores. The model directly generates graphs based on the RNN’s output at each time step.

---

### 4. **Attention Mechanism**
   - **With Attention (`train_rnn_epoch_with_attention`)**:
     - Explicit attention scores are used in the training process to modify the output prediction (`output_y_pred_step = output(output_x_step)`), and a sigmoid function is applied to the predictions to generate final outputs.
   
   - **Without Attention (`train_rnn_epoch`)**:
     - There is no mention or use of attention scores. The model simply applies a sigmoid function to the RNN's output: `y_pred = F.sigmoid(y_pred)`.

---


### Summary Table

| Feature                              | `train_rnn_epoch_with_attention`                                | `train_rnn_epoch`                                              |
|--------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------|
| **Attention Mechanism**              | Used in both training and testing.                               | Not used.                                                     |
| **Hidden State Handling**            | Modified by attention context vector.                           | Standard hidden state propagation without attention.           |
| **Forward Pass**                      | Attention scores and context vectors influence predictions.     | Direct use of RNN output without attention.                    |
| **Loss Calculation**                  | Involves attention-modified output.                             | Direct loss computation from the RNN output.                   |
| **Graph Generation**                  | Attention is involved in generating graphs.                     | Direct graph generation without attention.                     |
| **Output Handling**                   | Step-by-step prediction with attention.                         | Direct prediction without attention.                           |
| **Sequence Packing**                 | Packing handled with attention scores.                          | Standard packing and padding of sequences.                     |

---


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

