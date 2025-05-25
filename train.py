import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
import datetime
from utils import *
from model import *
from data import *
from args import Args
import create_graphs
import random
from graph_model import GraphGenModel
from torch_geometric.data import Data
from collections import deque

def approximate_community_loss(adj_matrix):
    """
    Approximates community structure using connected components.
    """
    G = nx.from_numpy_matrix(adj_matrix)
    num_nodes = G.number_of_nodes()
    components = list(nx.connected_components(G))
    num_components = len(components)
    community_score = num_components / num_nodes
    return abs(community_score - 0.25)  # Encourage moderate fragmentation

def compute_loss(output_seq, target_seq):
    # Assuming both output_seq and target_seq have shape [batch_size, seq_len, feature_dim]
    # Flatten both sequences to [batch_size * seq_len, feature_dim]
    output_seq = output_seq.reshape(-1, output_seq.size(-1))
    target_seq = target_seq.reshape(-1, target_seq.size(-1))
    
    # Compute Mean Squared Error loss
    loss = F.mse_loss(output_seq, target_seq)
    return loss
def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)

def gaussian_kernel_matrix(x: torch.Tensor,
                           y: torch.Tensor,
                           sigmas: torch.Tensor) -> torch.Tensor:
    """
    Computes a Gaussian RBF kernel matrix between x and y:
      K_ij = exp( -||x_i - y_j||^2 / (2 * sigma^2) )
    where we average over a *mixture* of bandwidths in `sigmas`.
    """
    # x: [n, d], y: [m, d]
    n, d = x.shape
    m, _ = y.shape

    # Compute squared distances ||x_i - y_j||^2 as a [n,m] matrix
    x_norm = (x**2).sum(dim=1).view(n, 1)  # [n,1]
    y_norm = (y**2).sum(dim=1).view(1, m)  # [1,m]
    dist2 = x_norm + y_norm - 2.0 * x @ y.t()  # [n,m]

    # Now apply each sigma
    sigmas = sigmas.view(-1, 1, 1)           # [S,1,1]
    beta = 1.0 / (2.0 * sigmas**2)           # [S,1,1]
    s = torch.exp(-beta * dist2.unsqueeze(0))  # [S,n,m]

    return s.mean(dim=0)  # average over sigmas → [n,m]


def mmd_rbf(x: torch.Tensor,
            y: torch.Tensor,
            sigmas: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the unbiased RBF‐MMD^2 between samples x and y.
    x: [n, d], y: [m, d].
    Returns: scalar MMD^2.
    """
    if sigmas is None:
        # default set of bandwidths (you can tweak these)
        sigmas = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0],
                              dtype=x.dtype,
                              device=x.device)

    # Compute kernels
    K_xx = gaussian_kernel_matrix(x, x, sigmas)
    K_yy = gaussian_kernel_matrix(y, y, sigmas)
    K_xy = gaussian_kernel_matrix(x, y, sigmas)

    # Unbiased estimator — drop diagonals for K_xx and K_yy
    n = x.size(0)
    m = y.size(0)
    # mask out diagonals
    mask_x = ~torch.eye(n, dtype=torch.bool, device=x.device)
    mask_y = ~torch.eye(m, dtype=torch.bool, device=y.device)

    mmd2 = (K_xx[mask_x].sum() / (n*(n-1))
           + K_yy[mask_y].sum() / (m*(m-1))
           - 2.0 * K_xy.sum() / (n*m))
    return mmd2

def bfs_order(edge_index, num_nodes, start=0):
    """
    Deterministic BFS over an undirected graph.
    edge_index may be:
      - a torch.Tensor of shape [2, E], or
      - a tuple/list (src_list, dst_list) of Python lists
    """
    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index
        pairs = zip(src.tolist(), dst.tolist())
    else:
        # assume edge_index == (src_list, dst_list)
        src_list, dst_list = edge_index
        pairs = zip(src_list, dst_list)

    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)

    # BFS
    order, seen, queue = [], {start}, deque([start])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return order

def preprocess_bfs(data: Data, max_prev_node: int):
    """
    1) Randomly permute node labels on data.edge_index’s device.
    2) Run BFS on the permuted graph.
    3) Build edge_seq in BFS order.
    """
    device = data.edge_index.device
    perm = torch.randperm(data.num_nodes, device=device)
    # remap edges via perm
    edge_idx = perm[data.edge_index]

    # Move to CPU for BFS
    src = edge_idx[0].cpu().tolist()
    dst = edge_idx[1].cpu().tolist()
    order = bfs_order((src, dst), data.num_nodes, start=0)

    T = len(order)
    edge_seq = torch.zeros((T, max_prev_node, 2), dtype=torch.long, device=device)
    seq_len  = torch.zeros(T, dtype=torch.long, device=device)

    for i, u in enumerate(order):
        prev = order[:i]
        L = min(len(prev), max_prev_node)
        seq_len[i] = L
        # reverse order as in GraphRNN
        for j in range(L):
            edge_seq[i, j, 0] = prev[-1-j]  # previous node
            edge_seq[i, j, 1] = u           # current node

    return edge_seq, seq_len

def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data[0]*feature_dim
    return loss_sum/(batch_idx+1)

def atest_rnn_dec_epoch(epoch, graphs_test, args, encoder, rnn, output, test_batch_size=16):
    # Initialize hidden states for both RNN and output
    rnn.hidden = rnn.init_hidden(test_batch_size, device='cuda')
    output.hidden = output.init_hidden(test_batch_size, device='cuda')
    
    rnn.eval()
    output.eval()

    # Generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # Discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

    # Assuming you have access to the encoder and a sample input for the encoder
    # Get a batch of test data (adjust this part based on your actual data loader)
    #dataset = Graph_sequence_sampler_pytorch(graphs_test, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node)

    # sampler = torch.utils.data.sampler.WeightedRandomSampler(
    #     [1.0 / len(dataset) for _ in range(len(dataset))],
    #     num_samples=args.batch_size * args.batch_ratio,
    #     replacement=True
    # )
    # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, collate_fn= custom_collate)
    test_batch = next(iter(graphs_test))
    x = test_batch['x'].to('cuda')
    edge_index = test_batch['edge_index'].to('cuda')
    batch = test_batch['batch'].to('cuda')
    print(f"Test batch keys: {test_batch.keys()}")
    print(f"x shape: {x.shape if x is not None else 'None'}")
    print(f"edge_index shape: {edge_index.shape if edge_index is not None else 'None'}")
    print(f"batch shape: {batch.shape if batch is not None else 'None'}")
    # Forward pass through the encoder to get graph embeddings
    encoder.eval()
    with torch.no_grad():
        graph_embedding = encoder(x, edge_index, batch).float()
    
    print(f"Graph Embedding Output Shape: {graph_embedding.shape}")  # Debugging
    # Normalize the graph embedding
    graph_embedding = (graph_embedding - graph_embedding.mean(dim=1, keepdim=True)) / (graph_embedding.std(dim=1, keepdim=True) + 1e-8)
    for i in range(max_num_node):
        batch_size = x_step.size(0)
        seq_len = x_step.size(1)
        #graph_embedding = torch.zeros(batch_size, 1, 128).to(x_step.device)  # Shape: [batch_size, 1, hidden_size]
        h = rnn(x_step, graph_embedding)
        #h = rnn(x_step)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)
        
        x_step = Variable(torch.zeros(test_batch_size, 1, args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).cuda()
        
        for j in range(min(args.max_prev_node, i + 1)):
            output_y_pred_step = output(output_x_step, graph_embedding=h)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:, :, j:j + 1] = output_x_step
            output.hidden = output.hidden.detach()
        
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = rnn.hidden.detach()
    
    y_pred_long_data = y_pred_long.data.long()

    # Save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # Get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def btest_dec_epoch(epoch, dataset_test, args, encoder, rnn, output, test_batch_size=16):
    encoder.eval()
    rnn.eval()
    output.eval()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G_pred_list = []

    with torch.no_grad():
        for batch_data in dataset_test:
            # Move batch data to GPU
            x = batch_data['x'].to(device)
            edge_index = batch_data['edge_index'].to(device)
            batch = batch_data['batch'].to(device)
            edge_seq = batch_data['edge_seq'].to(device).float()
            edge_lengths = batch_data['len']

            batch_size, seq_len, input_size = edge_seq.shape

            # Encode graph to get graph_embedding
            graph_embedding = encoder(x, edge_index, batch).float()
            hidden = rnn.init_hidden(graph_embedding)

            # Project edge_seq through rnn.input if rnn.has_input
            if rnn.has_input:
                edge_seq = edge_seq.view(-1, input_size)  # (B * T, 2)
                edge_seq = rnn.relu(rnn.input_raw(edge_seq))  # (B * T, emb)
                embedding_size = edge_seq.shape[-1]
                edge_seq = edge_seq.view(batch_size, seq_len, embedding_size)  # (B, T, emb)

            # Use first token as input
            input_t = edge_seq[:, 0, :].unsqueeze(1)  # (B, 1, emb)
            graph_embedding_step = graph_embedding.unsqueeze(1)

            output_dim = rnn.output[-1].out_features  # OR just infer from out_proj.size(-1)
            max_len = args.max_num_node - 1

            outputs = torch.zeros((batch_size, max_len, output_dim), device=device)
            # Autoregressive decoding
            for t in range(max_len):
                input_combined = torch.cat((input_t, graph_embedding_step), dim=-1)
                out, hidden = rnn.rnn(input_combined, hidden)
                out_proj = rnn.output(out)
                out_proj = torch.sigmoid(out_proj)

                outputs[:, t, :] = out_proj.squeeze(1)

                for t in range(max_len):
                    # No teacher forcing in test mode
                    input_t = rnn.relu(rnn.input_pred(out_proj.squeeze(1))).unsqueeze(1)

            hidden = hidden.detach()

            # Postprocess each sample in the batch
            for i in range(batch_size):
                adj_output = torch.tanh(outputs[i])
                adj_output = (adj_output + 1) / 2  # Normalize to [0, 1]
                adj_output = adj_output.cpu().numpy()

                # Decode adjacency matrix and convert to graph
                adj_pred = dec_decode_adj(adj_output)
                G_pred = get_graph(adj_pred)
                print(f"Decoded graph has {G_pred.number_of_nodes()} nodes, {G_pred.number_of_edges()} edges")
                G_pred_list.append(G_pred)

    return G_pred_list

def ctest_dec_epoch(epoch, dataset_test, args, encoder, rnn, output, test_batch_size=16):
    encoder.eval()
    rnn.eval()
    output.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G_pred_list = []

    max_num_node = int(args.max_num_node)
    max_prev_node = int(args.max_prev_node)

    with torch.no_grad():
        for batch_data in dataset_test:
            x = batch_data['x'].to(device)
            edge_index = batch_data['edge_index'].to(device)
            batch = batch_data['batch'].to(device)

            # Graph embedding
            graph_embedding = encoder(x, edge_index, batch).float()  # [batch_size, hidden_size]
            #print("[debug] Graph embedding sample:", graph_embedding[0].detach().cpu().numpy())
            hidden = rnn.init_hidden(graph_embedding)  # [num_layers, batch_size, hidden_size]

            # Init with <start> token
            x_step = torch.ones(test_batch_size, 1, max_prev_node).to(device)  # (B, 1, P)
            y_pred_long = torch.zeros(test_batch_size, max_num_node, max_prev_node).to(device)

            for i in range(max_num_node):
                # Input projection of raw step
                if rnn.has_input:
                    x_step_proj = rnn.relu(rnn.input(x_step.view(-1, max_prev_node)))  # [B, embed]
                    x_step_proj = x_step_proj.view(test_batch_size, 1, -1)  # [B, 1, embed]
                else:
                    x_step_proj = x_step  # Should already be projected correctly

                # Expand graph embedding for timestep
                graph_embedding_step = graph_embedding.unsqueeze(1)  # [B, 1, hidden]

                # Concatenate projected input and graph embedding
                rnn_input = torch.cat((x_step_proj, graph_embedding_step), dim=-1)  # [B, 1, embed + hidden]

                # GRU step
                out_proj, hidden = rnn.rnn(rnn_input, hidden)

                # Output projection (via output module)
                output_y_pred_step = output.output(out_proj)
                output_y_pred_step = torch.sigmoid(output_y_pred_step.squeeze(1))  # [B, P]
                print("[debug] Sigmoid output mean:", output_y_pred_step[0].mean().item())
                print("[debug] Sigmoid output max:", output_y_pred_step[0].max().item())
                print("[debug] Sigmoid output min:", output_y_pred_step[0].min().item())
                # Sampling
                output_x_step = (output_y_pred_step > 0.5).float()
                x_step_next = torch.zeros(test_batch_size, 1, max_prev_node).to(device)
                x_step_next[:, :, :output_x_step.shape[-1]] = output_x_step.unsqueeze(1)

                # Update inputs
                x_step = x_step_next
                y_pred_long[:, i:i + 1, :] = x_step

            # Convert and decode
            y_pred_long_data = y_pred_long.data.long()
            for i in range(test_batch_size):
                if i == 0:
                    
                    print("[debug] Raw decoder output:")
                    print(y_pred_long[0].cpu().numpy())  # shape (T, P)
                #print("Output sequence shape:", y_pred_long_data[i].shape)
                adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
                G_pred = get_graph(adj_pred)
                G_pred_list.append(G_pred)
                if i == 0:
                    print(f"Decoded graph has {G_pred.number_of_nodes()} nodes, {G_pred.number_of_edges()} edges")
    return G_pred_list


def atest_dec_unconditional(epoch, args, rnn, output, test_batch_size=16):
    """
    Autoregressive graph generation with the GRU_plain_dec decoder,
    *without* any dataset or encoder—just like test_rnn_epoch.
    """
    rnn.eval()
    output.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G_pred_list = []

    max_num_node  = int(args.max_num_node)
    max_prev_node = int(args.max_prev_node)

    with torch.no_grad():
        # 1) Create a dummy “graph_embedding” of zeros:
        #    shape = (batch_size, hidden_size)
        graph_embedding = torch.zeros(test_batch_size, rnn.hidden_size, device=device)
        hidden = rnn.init_hidden(graph_embedding)            # (num_layers, B, hidden_size)

        # 2) Initialize the first input token (<start>):
        x_step     = torch.ones(test_batch_size, 1, max_prev_node, device=device)
        y_pred_long = torch.zeros(test_batch_size, max_num_node, max_prev_node, device=device)

        # 3) Autoregressive loop
        for t in range(max_num_node):
            if rnn.has_input:
                # project the full “row” vector
                inp = x_step.view(-1, max_prev_node)        # (B, P)
                inp = rnn.relu(rnn.input_pred(inp))               # Linear(P→emb) + ReLU
                x_step_proj = inp.view(test_batch_size, 1, -1)
            else:
                x_step_proj = x_step

            # expand zero graph-embedding to concatenate
            graph_emb_step = graph_embedding.unsqueeze(1)    # (B, 1, hidden)
            rnn_in = torch.cat((x_step_proj, graph_emb_step), dim=-1)  # (B, 1, emb+hidden)

            out, hidden = rnn.rnn(rnn_in, hidden)            # one GRU step
            y_hat = torch.sigmoid(output.output(out).squeeze(1))  # (B, P)

            # sample a discrete row from the sigmoids
            row = sample_sigmoid(y_hat, sample=True, sample_time=1)  
            # pack it back to (B,1,P)
            x_step = row.unsqueeze(1)
            # stash it
            y_pred_long[:, t, :] = row

        # 4) Decode to networkx graphs
        y_long = y_pred_long.long().cpu().numpy()
        for i in range(test_batch_size):
            adj_pred = dec_decode_adj(y_long[i])
            G_pred   = get_graph(adj_pred)
            G_pred_list.append(G_pred)
            if i == 0:
                print(f"[uncond] Decoded graph has "
                      f"{G_pred.number_of_nodes()} nodes, "
                      f"{G_pred.number_of_edges()} edges")

    return G_pred_list

def btest_dec_unconditional(epoch, args, rnn, output, test_batch_size=16):
    """
    Autoregressive graph generation *without* any dataset or encoder.
    Uses squeeze/unsqueeze on x_step instead of view(), so you
    never have to worry about P or shape mismatches.
    """
    rnn.eval()
    output.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B = test_batch_size
    T = int(args.max_num_node)

    G_pred_list = []
    with torch.no_grad():
        # 1) zero “graph_embedding”
        H = rnn.hidden_size
        graph_embedding = torch.zeros(B, H, device=device)
        hidden = rnn.init_hidden(graph_embedding)  # (num_layers, B, H)

        # 2) start token: full-one row of length P = whatever input_pred expects
        P = output.output[0].in_features
        x_step      = torch.ones(B, 1, P, device=device)  # (B,1,P)
        y_pred_long = torch.zeros(B, T, P, device=device)

        for t in range(T):
            # squeeze off that middle dim → (B,P)
            inp = x_step.squeeze(1)

            # project (P → emb)
            inp_proj   = inp     # (B, emb)
            x_step_proj = inp_proj.unsqueeze(1)             # (B,1,emb)

            # concat with zero graph-embedding
            emb_step = graph_embedding.unsqueeze(1)         # (B,1,H)
            rnn_in   = torch.cat((x_step_proj, emb_step), dim=-1)

            # one GRU step + output
            out, hidden = rnn.rnn(rnn_in, hidden)
            y_hat = torch.sigmoid(output(out).squeeze(1))  # artık output.output değil  # (B,P)

            # sample next row
            row    = sample_sigmoid(y_hat, sample=True, sample_time=1)  # (B,P)
            x_step = row.unsqueeze(1)                           # (B,1,P)
            y_pred_long[:, t, :] = row

        # 3) decode into networkx graphs
        arr = y_pred_long.long().cpu().numpy()
        for i in range(B):
            adj = dec_decode_adj(arr[i])
            G   = get_graph(adj)
            G_pred_list.append(G)
            if i == 0:
                print(f"[uncond] Graph has {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges")

    return G_pred_list

def test_dec_unconditional(epoch, args, decoder, max_num_nodes=60, test_batch_size=16, device='cuda'):
    decoder.eval()
    
    # Start from a learned zero or fixed graph embedding
    graph_embedding = torch.zeros((1, test_batch_size, args.hidden_size_rnn), device=device)
    hidden = decoder.init_hidden(graph_embedding)

    # Dummy initial input (e.g., one-hot edge vector or learned start token)
    input_t = torch.zeros(test_batch_size, 1, args.embedding_size_rnn, device=device)
    outputs_edges = []
    outputs_values = []

    for t in range(max_num_nodes):
        graph_embedding_step = graph_embedding.permute(1, 0, 2)  # [B, 1, H]
        input_combined = torch.cat((input_t, graph_embedding_step), dim=-1)  # [B, 1, E+H]

        out, hidden = decoder.rnn(input_combined, hidden)
        edge_out = torch.sigmoid(decoder.output_edge(out))        # [B, 1, P]
        value_out = decoder.output_value(out)                     # [B, 1, D]

        outputs_edges.append(edge_out.squeeze(1))
        outputs_values.append(value_out.squeeze(1))

        # Use generated edge_out to produce next input
        input_t = decoder.relu(decoder.input_pred(edge_out.squeeze(1))).unsqueeze(1)

        # Optional: stopping condition — if all edge_out < threshold, assume generation done
        if (edge_out.squeeze(1) < 0.1).all(dim=1).any():
            break

    # Stack all outputs: [B, T, *]
    edge_tensor = torch.stack(outputs_edges, dim=1)
    value_tensor = torch.stack(outputs_values, dim=1)

    return edge_tensor, value_tensor

def atest_dec_epoch(args, encoder, decoder, data_loader=None):
    """
    Generate graphs from scratch using the trained encoder+decoder.
    Prints node and edge counts for each generated graph.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    decoder.eval()

    B = args.test_batch_size
    max_N = args.max_num_node
    E_prev = args.max_prev_node

    # We don’t need real data; just seed hidden with zeros
    # But to match GraphRNN, we can use encoder on a dummy batch if needed,
    # here we initialize hidden directly
    hidden = decoder.init_hidden(torch.zeros(B, decoder.hidden_size, device=device))

    # Placeholder for adjacency‐sequence predictions
    y_pred_long = torch.zeros(B, max_N, E_prev, device=device)

    # Start with an “all ones” previous‐node indicator for step 0
    x_step = torch.ones(B, 1, E_prev, device=device)

    for i in range(max_N):
        # Prepare a zero‐context embedding for unconditional generation:
        graph_embed_zero = torch.zeros(B, decoder.hidden_size, device=device)

        # Then call:
        edge_logits, _, hidden = decoder(
            x_step,             # sequence input, shape [B, 1, input_size=2]
            graph_embed_zero,   # graph_embedding, shape [B, hidden_size]
            hidden=hidden       # initial RNN hidden state
        )
        # unpack: since pack=False, logits come back directly
        #edge_logits = packed_logits  # shape [B, 1, E_prev]
        edge_probs  = torch.sigmoid(edge_logits.squeeze(1))  # [B, E_prev]
        edge_samples = (edge_probs > 0.5).float()

        # Place into y_pred_long
        y_pred_long[:, i, :] = edge_samples

        # Build next x_step: we need 1×E_prev for the next iteration
        # using the newly sampled edges, but shifted so only relevant positions
        next_x = torch.zeros_like(x_step)
        # For each position j< i, copy sample; else remain zero
        # Mask out positions j >= i
        mask = torch.arange(E_prev, device=device).unsqueeze(0) < (i+1)
        next_x[:, 0, :] = edge_samples * mask.float()
        x_step = next_x

    # Convert to CPU & numpy for decoding
    y_pred = y_pred_long.cpu().long().numpy()  # [B, max_N, E_prev]

    generated = []
    for b in range(B):
        # decode adjacency matrix
        adj = decode_adj(y_pred[b])
        G = nx.from_numpy_array(adj)
        generated.append(G)
        # print counts
        print(f"Graph {b:2d}: nodes = {G.number_of_nodes()}, edges = {G.number_of_edges()}")

    return generated

def btest_dec_epoch(args, encoder, decoder):
    import networkx as nx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder.eval()

    B = args.test_batch_size
    N = args.max_num_node
    M = args.max_prev_node

    # zero‐context embedding
    graph_zero = torch.zeros(B, decoder.hidden_size, device=device)
    hidden = decoder.init_hidden(graph_zero)  # [num_layers, B, H]

    # store predictions
    y_pred = torch.zeros(B, N, M, device=device)

    # for each new node i, build edge_seq of shape [B, i, 2]: pairs (j, i)
    for i in range(N):
        if i == 0:
            edge_seq = torch.zeros(B, 1, 2, device=device, dtype=torch.float)
            # (no previous nodes → dummy zeros)
        else:
            # create batch of [i x 2] index pairs [(0,i), (1,i), ..., (i-1,i)]
            prev_idx = torch.arange(i, device=device).unsqueeze(1)   # [i,1]
            curr_idx = torch.full((i,1), i, device=device)         # [i,1]
            pair = torch.cat([prev_idx, curr_idx], dim=1).float()   # [i,2]
            # repeat for batch:
            edge_seq = pair.unsqueeze(0).repeat(B, 1, 1)            # [B,i,2]

        # forward step (no pack):
        logits, _, hidden = decoder(edge_seq, graph_zero, hidden=hidden)
        # logits: [B, i, M]? No—decoder ignores sequences > M; but here i ≤ M
        # Take last timestep only (node i) if decoder returns full sequence:
        # For GRU_flat_dec_multihead, it returns all timesteps → we need the last:
        node_logits = logits[:, -1, :]           # [B, M]
        probs = torch.sigmoid(node_logits)
        samples = sample_sigmoid(probs, sample=True, sample_time=1)

        y_pred[:, i, :] = samples

    # build graphs
    generated = []
    for b in range(B):
        adj = decode_adj(y_pred[b].cpu().long().numpy())
        G = nx.from_numpy_matrix(adj)
        print(f"Graph {b+1}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        generated.append(G)

    return generated

def ctest_dec_epoch(args, encoder, decoder, dataset_test=None):
    import networkx as nx

    device = next(decoder.parameters()).device
    decoder.eval()

    B = args.test_batch_size
    N = args.max_num_node
    M = args.max_prev_node
    temp = 0.7

    generated = []

    for b in range(B):
        # ---- seed latent ----
        if dataset_test:
            # conditional: use real graph embed
            x, edge_index, batch = dataset_test[b]
            graph_embed = encoder(x.to(device), edge_index.to(device), batch.to(device)).unsqueeze(0)
        else:
            # unconditional: random normal
            graph_embed = torch.randn(1, decoder.hidden_size, device=device)
        hidden = decoder.init_hidden(graph_embed)  # [L,1,H]

        # ---- generate BFS order ----
        adj_pred = np.zeros((N,N))
        for i in range(N):
            # build edge_seq in BFS order: previous BFS nodes to current i
            if i==0:
                edge_seq = torch.zeros(1,1,2,device=device)
            else:
                # BFS parents: here simply previous indices (in BFS you'd have parent list)
                prev = torch.arange(i, device=device).unsqueeze(1).float()
                curr = torch.full((i,1), i, device=device).float()
                edge_seq = torch.cat([prev, curr], dim=1).unsqueeze(0)  # [1,i,2]

            # forward
            logits, _, hidden = decoder(edge_seq, graph_embed, hidden=hidden)
            # sample
            probs = torch.sigmoid(logits.squeeze(1) / temp)  # [1,i,M]
            samples = torch.bernoulli(probs).detach().cpu().numpy().astype(int)[0]

            # write into adj_pred using BFS mapping
            for j, val in enumerate(samples):
                if val.item() == 1 and j < i:
                    adj_pred[i, j] = 1

        # symmetrize
        adj_full = adj_pred + adj_pred.T
        G = nx.from_numpy_matrix(adj_full)
        generated.append(G)
        print(f"Graph {b+1}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    return generated



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def test_rnn_with_attention_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h, hidden_state = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        # Handle single-layer and multi-layer RNN cases
        if hidden_state.dim() == 2:  # Single-layer RNN
            hidden_state = hidden_state.unsqueeze(0)  # Add layer dimension: (1, batch_size, hidden_size)
            hidden_null = torch.zeros(
                args.num_layers - 1, hidden_state.size(1), hidden_state.size(2)
            ).cuda()
        else:  # Multi-layer RNN
            hidden_null = torch.zeros(
                args.num_layers - 1, hidden_state.size(1), hidden_state.size(2)
            ).cuda()
        #hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((hidden_state, hidden_null), dim=0)
        #output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
        #                           dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid_attention(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list



def train_rnn_epoch_with_attention(epoch, args, rnn, output, data_loader,
                                   optimizer_rnn, optimizer_output,
                                   scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        
        # Extract data
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        # Initialize RNN hidden state
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # Sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        
        # Pack the padded sequence
        packed_y = pack_padded_sequence(y, y_len, batch_first=True, enforce_sorted=False).cuda()

        # Now reverse y_reshape (i.e., the data) according to lengths
        idx = [i for i in range(packed_y.data.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        y_reshape = packed_y.data.index_select(0, idx)

        # Add extra dimension
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        # Define output_y
        output_y = y_reshape  # Ensure output_y is assigned here

        # Prepare input lengths for packing
        output_y_len = np.array(y_len)  # y_len should be the sequence lengths of each sample in the batch

        # Ensure the sequence lengths do not exceed y.size(2)
        output_y_len = np.clip(output_y_len, 0, y.size(2))

        # Ensure that output_y_len is a tensor of length batch_size
        output_y_len = torch.tensor(output_y_len)  # Convert to tensor for compatibility with pack_padded_sequence
        # Convert to Variables and move to GPU
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        
        # Unpack the packed sequence
        unpacked_y, lengths = pad_packed_sequence(packed_y, batch_first=True)
        # Prepare tensors for the output module
        ones_tensor = torch.ones(unpacked_y.size(0), 1, 1).cuda()
        #print(f"Shape of ones_tensor: {ones_tensor.shape}")
        #print(f"Shape of unpacked: {unpacked_y.shape}")
        output_x = torch.cat((ones_tensor, unpacked_y[:, :-1, :1]), dim=1)
        
        #print(output_x.shape)
        output_x = Variable(output_x).cuda()
        output_y = Variable(unpacked_y).cuda()
        
        # Forward pass through RNN with attention
        context_vector, attention_scores = rnn(x, pack=True, input_len=y_len)

        # Create hidden_null for remaining layers
        #print(h_data.shape)
        num_layers = rnn.hidden.size(0)
        hidden_null = torch.zeros(num_layers - 1, context_vector.size(1)).cuda()

        
        # Combine h_data and hidden_null
        output.hidden = torch.cat((context_vector, hidden_null), dim=0)


        
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Extract the output tensor
        print(y_pred.shape)
        # Apply sigmoid to the correct tensor
        y_pred = torch.sigmoid(y_pred)
        print(y_pred.shape)

        # # Ensure y_pred has the correct dimensions
        # if y_pred.ndimension() == 2:  # If (batch_size, feature_dim)
        #             y_pred = y_pred.unsqueeze(1)  # Add sequence dimension -> (batch_size, 1, feature_dim)

        # seq_len = max(output_y_len)  # Get the maximum sequence length from the output
        # if y_pred.shape[1] < seq_len:
        #     padding = seq_len - y_pred.shape[1]
        #     y_pred = F.pad(y_pred, (0, 0, 0, padding))  # Pad sequence dimension (2nd dimension)
        # elif y_pred.shape[1] > seq_len:
        #     y_pred = y_pred[:, :seq_len, :]  # Truncate sequence dimension to match
        # # Expand y_pred to match output_y dimensions
        # y_pred = y_pred.expand(-1, -1, output_y.size(2))
        #y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True, enforce_sorted=False)
        #output_y_packed = pack_padded_sequence(output_y, output_y_len, batch_first=True, enforce_sorted=False)
        if y_pred.ndimension() == 2:
            y_pred = y_pred.unsqueeze(1)  # Add seq_len dimension
        seq_len = max(output_y_len)
        if y_pred.size(1) < seq_len:
            padding = seq_len - y_pred.size(1)
            y_pred = F.pad(y_pred, (0, 0, 0, padding))  # Pad seq_len dimension
        
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # Debug output shapes
        #print(f"Shape of y_pred before unpacking: {y_pred.shape}")
        #print(f"Shape of output_y before unpacking: {output_y.shape}")
        
        # If unpacking is needed
        # y_pred, _ = pad_packed_sequence(y_pred_packed, batch_first=True)
        # output_y, _ = pad_packed_sequence(output_y_packed, batch_first=True)

        # Debug output shapes
        #print(f"Shape of y_pred after unpacking: {y_pred.shape}")
        #print(f"Shape of output_y after unpacking: {output_y.shape}")
        # Compute loss
        y_pred_ = y_pred.expand(-1, -1, output_y.size(2))  # Expand last dimension to match output_y
        loss = binary_cross_entropy_weight(y_pred_, output_y)
        #print(loss)
        loss.backward()

        # Apply gradient clipping
        #torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=5.0)

        # Update optimizers and schedulers
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        # Log training progress
        if  batch_idx == 0: #epoch % args.epochs_log == 0 and
            print(f'Epoch: {epoch}/{args.epochs}, '
                f'train loss: {loss.item():.6f}, '
                f'graph type: {args.graph_type}, '
                f'num_layer: {args.num_layers}, '
                f'hidden: {args.hidden_size_rnn}')

        # Logging
        log_value(f'loss_{args.fname}', loss.item(), epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim

    return loss_sum / (batch_idx + 1)


def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        

        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)

def arnn_forward_train(
    rnn, output_proj, graph_embedding, edge_seq, raw_edge_seq, y,
    edge_lengths, max_len, teacher_forcing_ratio, device
):
    batch_size, seq_len, input_size = edge_seq.size()
    output_dim = y.size(-1)

    hidden = rnn.init_hidden(graph_embedding)
    graph_embedding_step = graph_embedding.unsqueeze(1)

    if rnn.has_input:
        edge_seq = edge_seq.view(-1, input_size)
        edge_seq = rnn.relu(rnn.input_raw(edge_seq))
        emb_size = edge_seq.size(-1)
        edge_seq = edge_seq.view(batch_size, seq_len, emb_size)

    input_t = edge_seq[:, 0, :].unsqueeze(1)
    outputs = torch.zeros((batch_size, max_len, output_dim), device=device)

    for t in range(max_len):
        input_combined = torch.cat((input_t, graph_embedding_step), dim=-1)
        out, hidden = rnn.rnn(input_combined, hidden)
        out_proj = torch.sigmoid(output_proj(out))
        outputs[:, t, :] = out_proj.squeeze(1)

        teacher_force = random.random() < teacher_forcing_ratio
        if t + 1 < max_len:
            if teacher_force:
                raw_edge_input = raw_edge_seq[:, t + 1, :]
                input_t = rnn.relu(rnn.input_raw(raw_edge_input)).unsqueeze(1)
            else:
                input_t = rnn.relu(rnn.input_pred(out_proj.squeeze(1))).unsqueeze(1)

    return outputs[:, :y.size(1), :], hidden

def rnn_forward_train(
    rnn, output_proj, graph_embedding, edge_seq, raw_edge_seq, y,
    edge_lengths, max_len, teacher_forcing_ratio, device
):
    batch_size, seq_len, input_size = edge_seq.size()
    output_dim = y.size(-1)

    hidden = rnn.init_hidden(graph_embedding)
    graph_embedding_step = graph_embedding.unsqueeze(1)

    if rnn.has_input:
        edge_seq = edge_seq.view(-1, input_size)
        edge_seq = rnn.relu(rnn.input_raw(edge_seq))
        emb_size = edge_seq.size(-1)
        edge_seq = edge_seq.view(batch_size, seq_len, emb_size)

    input_t = edge_seq[:, 0, :].unsqueeze(1)  # [B, 1, E]
    outputs = torch.zeros((batch_size, max_len, output_dim), device=device)

    for t in range(max_len):
        input_combined = torch.cat((input_t, graph_embedding_step), dim=-1)  # [B, 1, E+H]
        out, hidden = rnn.rnn(input_combined, hidden)
        out_proj = torch.sigmoid(output_proj(out))
        outputs[:, t, :] = out_proj.squeeze(1)

        if t + 1 < seq_len:
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                raw_edge_input = raw_edge_seq[:, t + 1, :]  # [B, P]
                input_t = rnn.relu(rnn.input_raw(raw_edge_input)).unsqueeze(1)
            else:
                input_t = rnn.relu(rnn.input_pred(out_proj.squeeze(1))).unsqueeze(1)
        else:
            input_t = rnn.relu(rnn.input_pred(out_proj.squeeze(1))).unsqueeze(1)

    # Apply mask to zero out padded positions in the loss (optional)
    # Mask: 1 where length > t
    edge_lengths = edge_lengths.to(device)  # Add this before using edge_lengths
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < edge_lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(outputs)

    outputs = outputs * mask  # apply mask to output (optional)
    return outputs[:, :y.size(1), :], hidden


def rnn_forward_train_multihead(
    decoder, graph_embedding, edge_seq, raw_edge_seq, edge_targets, value_targets,
    edge_lengths, max_len, teacher_forcing_ratio, device
):
    batch_size, seq_len, input_size = edge_seq.size()
    hidden = decoder.init_hidden(graph_embedding)
    graph_embedding_step = graph_embedding.unsqueeze(1)

    # Project initial input
    edge_seq = edge_seq.view(-1, input_size)
    edge_seq = decoder.relu(decoder.input(edge_seq))
    emb_size = edge_seq.size(-1)
    edge_seq = edge_seq.view(batch_size, seq_len, emb_size)

    input_t = edge_seq[:, 0, :].unsqueeze(1)
    edge_outputs = torch.zeros((batch_size, max_len, edge_targets.size(-1)), device=device)
    value_outputs = torch.zeros((batch_size, max_len, value_targets.size(-1)), device=device)

    for t in range(max_len):
        input_combined = torch.cat((input_t, graph_embedding_step), dim=-1)
        out, hidden = decoder.rnn(input_combined, hidden)

        # Multi-head projection
        edge_out = torch.sigmoid(decoder.output_edge(out))     # Edge logits
        value_out = decoder.output_value(out)                  # Value predictions (regression or classification)

        edge_outputs[:, t, :] = edge_out.squeeze(1)
        value_outputs[:, t, :] = value_out.squeeze(1)

        if t + 1 < seq_len:
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                next_input = raw_edge_seq[:, t + 1, :]
                input_t = decoder.relu(decoder.input(next_input)).unsqueeze(1)
            else:
                input_t = decoder.relu(decoder.input(edge_out.squeeze(1))).unsqueeze(1)
        else:
            input_t = decoder.relu(decoder.input(edge_out.squeeze(1))).unsqueeze(1)

    # Masking
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < edge_lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)

    edge_outputs = edge_outputs * mask
    value_outputs = value_outputs * mask

    return edge_outputs[:, :edge_targets.size(1), :], value_outputs[:, :value_targets.size(1), :], hidden

def new_train(model, train_loader, optimizer, epochs, device):
    """
    Train a GraphRNN-inspired model with a GNN encoder and GRU decoder.
    Assumes train_loader yields batches with .x, .edge_index, .batch, .edge_seq, .y attributes.
    Uses the weighted BCE loss function provided (binary_cross_entropy_weight).
    Logs average loss per epoch.
    """
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in train_loader:
            # Move batch data to device
            x = batch.x.to(device)                 # Node feature matrix [N, 80]
            edge_index = batch.edge_index.to(device)  # Graph connectivity [2, E]
            batch_idx = batch.batch.to(device)     # Batch indices [N]
            edge_seq = batch.edge_seq.to(device)   # Edge input sequence [B, 247, 2]
            y_target = batch.y.to(device)          # Node feature targets [B, 55, 80]
            
            optimizer.zero_grad()
            # Encode graph to get initial hidden state for the decoder
            # Assume encoder returns [batch_size, hidden_dim]
            hidden = model.encoder(x, edge_index, batch_idx)  
            # Prepare GRU hidden: (num_layers, batch, hidden_dim). Here 1 layer is assumed.
            hidden = hidden.unsqueeze(0)  # shape (1, B, hidden_dim)

            # Decode edges and values with teacher forcing
            edge_logits, value_preds = model.decoder(edge_seq, hidden)
            # edge_logits: [B, 247, 2], value_preds: [B, 55, 80]

            # Compute weighted BCE loss on both outputs&#8203;:contentReference[oaicite:3]{index=3}
            loss_edges = binary_cross_entropy_weight(edge_logits, edge_seq)
            loss_values = binary_cross_entropy_weight(value_preds, y_target)
            loss = loss_edges + loss_values

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # Log average loss for this epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

def agenerate_graphs(model, num_graphs=1, max_nodes=55, device='cpu'):
    """
    Generate graphs by autoregressively sampling edges and node features.
    Returns a list of NetworkX Graphs with node attribute 'feature' set.
    """
    model.eval()
    generated_graphs = []
    for _ in range(num_graphs):
        with torch.no_grad():
            # Initialize hidden state (use zeros or learned init; here assume zeros)
            # Assuming model.encoder.hidden_size exists, or infer from model
            hidden_dim = model.encoder.hidden_dim if hasattr(model.encoder, 'hidden_dim') else model.decoder.hidden_size
            hidden = torch.zeros(1, 1, hidden_dim, device=device)

            G = nx.Graph()
            # Iteratively add nodes
            for node_i in range(max_nodes):
                # Prepare dummy edge-sequence input; exact form depends on model API
                # Here we use a zero tensor of shape (1, 1, 2) as a placeholder.
                if node_i == 0:
                    edge_input = torch.zeros((1, 1, 2), device=device)
                else:
                    # Optionally, use previously sampled edges as input; this depends on your decoder
                    edge_input = torch.zeros((1, 1, 2), device=device)

                # Decode one step: outputs for current node
                edge_logits, value_logit = model.decoder(edge_input, hidden)
                # edge_logits: shape [1, 247, 2] (or relevant shape for edges), value_logit: [1,1,80] or [1,80]
                edge_logits = edge_logits.view(-1)  # flatten if needed

                # Sample edges: assume edges to nodes [0..node_i-1] are in the first node_i entries
                edge_probs = torch.sigmoid(edge_logits[:node_i])  # take as many as existing nodes
                sampled_edges = (edge_probs > 0.5).cpu().numpy().astype(int)

                # Add edges for this new node
                for prev in range(node_i):
                    if sampled_edges[prev] == 1:
                        G.add_edge(node_i, prev)

                # Sample node feature vector
                value_probs = torch.sigmoid(value_logit).squeeze()  # shape [80]
                sampled_feat = (value_probs > 0.5).cpu().numpy().astype(int)

                # Add new node with feature attribute
                G.add_node(node_i, feature=sampled_feat)

                # (Optional) update hidden state if decoder returns new hidden (not shown here)
                # hidden = new_hidden

            generated_graphs.append(G)
    return generated_graphs

def generate_graphs(args, encoder, decoder, num_graphs=1, max_num_nodes=55, device='cpu'):
    """
    Autoregressively generate graphs using the trained decoder (GRU_flat_dec_multihead).
    Encoder is unused during generation unless you use context-conditional generation.
    """
    decoder.eval()
    generated_graphs = []

    for _ in range(num_graphs):
        with torch.no_grad():
            # Start with zero hidden state
            hidden_dim = decoder.hidden_size
            hidden = torch.zeros(decoder.num_layers, 1, hidden_dim, device=device)

            G = nx.Graph()

            for node_i in range(max_num_nodes):
                # Build input edge sequence: shape [1, 1, 2] or context-dependent
                # Here: feed [prev_node_id, curr_node_id] style dummy input
                if node_i == 0:
                    edge_input = torch.zeros((1, 1, 2), device=device)
                else:
                    # Create a dummy edge sequence of size [1, node_i, 2]
                    edge_input = torch.zeros((1, node_i, 2), device=device)
                    for j in range(node_i):
                        edge_input[0, j, 0] = j
                        edge_input[0, j, 1] = node_i

                # Decoder forward
                edge_logits, value_logit, _ = decoder(edge_input, hidden)

                # Sample edges
                edge_probs = torch.sigmoid(edge_logits[0, :node_i, 0])  # shape [node_i]
                sampled_edges = (edge_probs > 0.5).cpu().numpy().astype(int)

                # Add node and edges
                G.add_node(node_i)
                for j in range(node_i):
                    if sampled_edges[j] == 1:
                        G.add_edge(node_i, j)

                # Sample node feature (if applicable)
                if value_logit is not None:
                    value_probs = torch.sigmoid(value_logit).squeeze(0).squeeze(0)  # shape [80]
                    sampled_feat = (value_probs > 0.5).cpu().numpy().astype(int)
                    G.nodes[node_i]['feature'] = sampled_feat

            generated_graphs.append(G)

    return generated_graphs





def train_attention_epoch(epoch, args, encoder, decoder, data_loader,
                          opt_enc, opt_dec, sched_enc, sched_dec):
    encoder.train(); decoder.train()
    total_loss, total_features = 0, 0

    for batch_idx, data in enumerate(data_loader):
        # --- Prepare inputs & targets ---
        x = data['x'].float().cuda()                     # [B, T, F]
        y = data['y'].float().cuda()                     # [B, T, F]
        y_len = data['len']                              # list of lengths
        T_max = max(y_len)
        x, y = x[:, :T_max], y[:, :T_max]

        # --- Encoder forward ---
        enc_hidden = encoder.init_hidden(x.size(0))
        packed_x = pack_padded_sequence(x, y_len, batch_first=True, enforce_sorted=False)
        enc_out_packed, enc_hidden = encoder.gru(packed_x, enc_hidden)   # :contentReference[oaicite:3]{index=3}
        encoder_outputs, _ = pad_packed_sequence(enc_out_packed, batch_first=True)
        # Build mask: 1 for valid timesteps, 0 for padding
        mask = (torch.arange(T_max)[None, :].cuda() < torch.tensor(y_len)[:, None].cuda()).long()

        # --- Initialize decoder ---
        dec_hidden = enc_hidden                                         # use encoder final state
        batch, seq, feat = y.size()
        # Start tokens: ones like original
        dec_input = torch.ones(batch, 1, feat).cuda()

        # Zero grads
        opt_enc.zero_grad(); opt_dec.zero_grad()

        # --- Step through decoder with attention ---
        loss = 0
        for t in range(seq):
            # Get true target at this step
            y_t = y[:, t:t+1, :]

            # Decoder with attention returns logits, next hidden, attn
            logits, dec_hidden, _ = decoder(dec_input, dec_hidden, encoder_outputs, mask)
            logits = torch.sigmoid(logits)                              # [B,1,F]

            # Compute BCE loss for this step
            loss += F.binary_cross_entropy(logits, y_t, reduction='sum')  # :contentReference[oaicite:4]{index=4}

            # Prepare next input (teacher forcing)
            dec_input = y_t

        # --- Backprop & optimize ---
        loss.backward()
        opt_enc.step(); opt_dec.step()
        sched_enc.step(); sched_dec.step()

        # --- Logging ---
        feature_dim = seq * feat
        total_loss += loss.item()
        total_features += feature_dim
        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print(f'Epoch {epoch}/{args.epochs}, Loss: {loss.item()/feature_dim:.6f}')

    return total_loss / total_features


def test_attention_epoch(args, encoder, decoder):
    encoder.eval(); decoder.eval()
    B = args.test_batch_size
    max_nodes = args.max_num_node
    max_prev = args.max_prev_node

    # Prepare a dummy input of all-ones for initial step
    dec_input = torch.ones(B, 1, max_prev).cuda()
    enc_hidden = encoder.init_hidden(B)

    # No ground truth; we generate sequentially
    # First, dummy encoder_outputs = zeros (or could encode prior graph context)
    encoder_outputs = torch.zeros(B, 1, args.hidden_size_rnn).cuda()
    mask = torch.ones(B, 1).long().cuda()

    # Collect long predictions
    y_pred_long = []

    # Iterate nodes
    for i in range(max_nodes):
        # One decode step
        logits, enc_hidden, _ = decoder(dec_input, enc_hidden, encoder_outputs, mask)
        probs = torch.sigmoid(logits).squeeze(1)                # [B, F]
        # Sample or argmax
        dec_input = (probs > 0.5).float().unsqueeze(1)         # [B,1,F]
        y_pred_long.append(dec_input)

        # Optionally update encoder_outputs/ mask if using dynamic context
        # (for simplicity, we keep fixed zero context here)

    # Stack predictions: [B, max_nodes, F]
    y_pred_long = torch.cat(y_pred_long, dim=1).long().cpu().numpy()

    # Decode to graph objects
    G_list = []
    for i in range(B):
        adj = decode_adj(y_pred_long[i])
        G_list.append(get_graph(adj))

    return G_list







def train(args, dataset_train, encoder, decoder):
    import tqdm

    # Check if loading existing model
    epoch = 1
    if args.load:
        encoder_path = args.model_save_path + args.fname + 'encoder_' + str(args.load_epoch) + '.dat'
        decoder_path = args.model_save_path + args.fname + 'decoder_' + str(args.load_epoch) + '.dat'

        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
        args.lr = 0.00001
        epoch = args.load_epoch + 1
        print(f'Models loaded from epoch {args.load_epoch}, lr set to {args.lr}')

    # Move models to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)

    # Optimizers and schedulers
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler_encoder = MultiStepLR(optimizer_encoder, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_decoder = MultiStepLR(optimizer_decoder, milestones=args.milestones, gamma=args.lr_rate)

    # Timing info
    time_all = np.zeros(args.epochs)

    for epoch in tqdm.trange(epoch, args.epochs + 1, desc="Training Epochs"):
        now = datetime.datetime.now()
        print(f"\n Epoch {epoch} started at {now.time()}")

        start_time = tm.time()

        if 'GraphRNN_RNN' in args.note:
            # assumes you’ve implemented train_attention_epoch exactly like your
            # original train_rnn_epoch signature, but invoking attention
            # train_attention_epoch(
            # epoch, args,
            # encoder, decoder,
            # dataset_train,
            # optimizer_encoder, optimizer_decoder,   # pass both optimizers
            # scheduler_encoder, scheduler_decoder    # pass both schedulers
            # )
            train_rnn_epoch(epoch, args, encoder, decoder, dataset_train,
                            optimizer_encoder, optimizer_decoder, 
                            scheduler_encoder, scheduler_decoder
                            )
        else:
            train_dec_epoch(
                epoch, args, encoder, decoder, dataset_train,
                optimizer_encoder, optimizer_decoder,
                scheduler_encoder, scheduler_decoder
            )

        end_time = tm.time()
        time_all[epoch - 1] = end_time - start_time
        # Test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            G_pred = []
            while len(G_pred) < args.test_total_size:
                if 'GraphRNN_ATT' in args.note:
                    G_pred_step = test_attention_epoch(args, encoder, decoder)
                else:    
                    G_pred_step = test_dec_epoch(args, encoder, decoder, dataset_train)
                G_pred.extend(G_pred_step)

            # Save graphs
            fname = args.graph_save_path + args.fname_pred + str(epoch) + '.dat'
            save_graph_list(G_pred, fname)
            print('Test done, graphs saved.')

        # Save model
        if args.save and epoch % args.epochs_save == 0:
            encoder_path = args.model_save_path + args.fname + 'encoder_' + str(epoch) + '.dat'
            decoder_path = args.model_save_path + args.fname + 'decoder_' + str(epoch) + '.dat'
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)

    np.save(args.timing_save_path + args.fname, time_all)


def train_dec_epoch(epoch, args, encoder, decoder, data_loader,
                    optimizer_enc, optimizer_dec,
                    scheduler_enc, scheduler_dec):
    """
    Trains one epoch of the decoder, augmenting the BCE edge loss
    with a soft‐modularity auxiliary loss and an RBF‐MMD degree loss.
    """
    encoder.train()
    decoder.train()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_elems = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for batch_idx, data in enumerate(data_loader):
        # 1) Load & move to device
        x_all      = data['x'].float().to(device)
        edge_index = data['edge_index'].to(device)
        batch_all  = data['batch'].to(device)
        edge_seq   = data['edge_seq'].float().to(device)
        y          = data['y'].float().to(device)       # [B, N, max_prev]
        node_lens  = data['len']                       # [B]

        # 2) Encode graph to get embedding
        graph_embed = encoder(x_all, edge_index, batch_all)  # [B, H]

        # 3) Init decoder hidden
        hidden = decoder.init_hidden(graph_embed)            # [L, B, H]

        # 4) Decode in packed mode
        packed_logits, _, _ = decoder(
            edge_seq, graph_embed,
            hidden=hidden, pack=True, input_len=node_lens
        )

        # 5) Unpack logits to dense [B, N_true, max_prev]
        edge_logits, _ = pad_packed_sequence(packed_logits, batch_first=True)
        B, N_pred, max_prev = edge_logits.size()
        N_true = y.size(1)
        edge_logits = edge_logits[:, :N_true, :]

        # 6) Mask for valid (i,j)
        lens_tensor = torch.tensor(node_lens, device=device)
        mask_rows = torch.arange(N_true, device=device).unsqueeze(0) < lens_tensor.unsqueeze(1)
        mask3 = mask_rows.unsqueeze(-1).expand(-1, -1, max_prev)

        # 7) BCE loss
        logits_flat  = edge_logits[mask3]
        targets_flat = y[mask3]
        loss_bce     = bce_loss(logits_flat, targets_flat)
        loss = loss_bce

        # --- build P_full for auxiliary losses ---
        if args.lambda_mod > 0 or getattr(args, 'lambda_deg', 0) > 0:
            # 8a) Soft adjacency P_full [B, N_true, N_true]
            P_full = torch.zeros(B, N_true, N_true, device=device)
            for i in range(N_true):
                prev_len = min(i, max_prev)
                if prev_len > 0:
                    probs = torch.sigmoid(edge_logits[:, i, :prev_len])  # [B, prev_len]
                    P_full[:, i, :prev_len] = probs
            # 8b) Symmetrize
            P = P_full + P_full.transpose(1,2)  # [B, N, N]

        # 8) Modularity loss
        if getattr(args, 'lambda_mod', 0) > 0:
            k = P.sum(dim=-1)                  # [B, N]
            m = k.sum(dim=-1) * 0.5            # [B]
            kk = k.unsqueeze(-1) * k.unsqueeze(-2)     # [B, N, N]
            B_mat = P - kk.div(2 * m.view(-1,1,1))
            Q = (B_mat * P).sum(dim=(-2,-1)).div(2 * m)  # [B]
            loss_mod = - Q.mean()
            loss = loss + args.lambda_mod * (loss_mod)

        # 9) Degree‐MMD loss
        if getattr(args, 'lambda_deg', 0) > 0:
            # flatten batch of soft‐degrees vs true degrees
            deg_pred = P.sum(dim=-1).view(-1, 1)            # [B*N,1]
            # compute true adjacency A from y
            A = torch.zeros_like(P)
            for i in range(N_true):
                prev_len = min(i, max_prev)
                if prev_len > 0:
                    A[:, i, :prev_len] = y[:, i, :prev_len]
            A = A + A.transpose(1,2)
            deg_true = A.sum(dim=-1).view(-1, 1)           # [B*N,1]

            loss_deg = mmd_rbf(deg_pred, deg_true)
            loss = loss + args.lambda_deg * loss_deg

        # 10) Backprop / step
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
        scheduler_enc.step()
        scheduler_dec.step()

        # 11) Track
        total_loss  += loss.item() * lens_tensor.sum().item()
        total_elems += lens_tensor.sum().item()

        if batch_idx == 0 and epoch % args.epochs_log == 0:
            print(f"[Epoch {epoch}/{args.epochs}] "
                  f"BCE={loss_bce.item():.4f}  "
                  f"Mod={loss_mod.item() if args.lambda_mod>0 else 0:.4f}  "
                  f"MMDdeg={loss_deg.item() if args.lambda_deg>0 else 0:.4f}  "
                  f"Total={loss.item():.4f}")

    return total_loss / total_elems

def test_dec_epoch(args, encoder, decoder, data_loader):
    """
    Uses each graph’s true encoder embedding to initialize the decoder,
    then generates autoregressively.  We (1) sample edges via sigmoid+0.5,
    and (2) build the NetworkX graph via from_numpy_array to include isolates.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    decoder.eval()

    G_pred = []
    total  = args.test_total_size
    maxN, maxP = args.max_num_node, args.max_prev_node

    with torch.no_grad():
        for data in data_loader:
            # 1) encode real graphs
            x_all      = data['x'].float().to(device)
            edge_index = data['edge_index'].to(device)
            batch_all  = data['batch'].to(device)
            graph_embed = encoder(x_all, edge_index, batch_all)  # [B, H]
            graph_embed_zero = torch.zeros_like(graph_embed)
            B, _        = graph_embed.size()
            lengths = data['len'].tolist()   # true node counts per graph
            # 2) decode each graph in batch
            for b in range(B):
                if len(G_pred) >= total:
                    break

                embed_b = graph_embed[b:b+1]               # [1, H]
                hidden  = decoder.init_hidden(embed_b)     # [L, 1, H]

                # prepare storage for adjacency‑vector predictions
                length_b = lengths[b]
                y_pred = torch.zeros(1, maxN, maxP, device=device)
                # 3) autoregressive edge generation
                for i in range(maxN):
                    # build the same (prev, curr) pairs used at training time
                    if i == 0:
                        edge_seq = torch.zeros(1, 1, 2, device=device)
                    else:
                        prev = torch.arange(i, device=device).view(i,1).float()
                        curr = torch.full((i,1), i, device=device).float()
                        edge_seq = torch.cat([prev, curr], dim=1).unsqueeze(0)

                    # get logits from decoder
                    logits, _, hidden = decoder(edge_seq, embed_b,
                                                hidden=hidden, pack=False)
                    last_logits = logits[:, -1, :]           # [1, max_prev]
                    t = 0.3  # lower = sharper
                    sharp_logits = last_logits / t
                    # --- key change: use plain sigmoid + 0.5 threshold ---
                    probs  = torch.sigmoid(sharp_logits)      # logistic sigmoid :contentReference[oaicite:3]{index=3}
                    ###Enes
                    #p_pos = 0.7   # keep fairly strong intra-block edges
                    p_pos = 0.9   # keep fairly strong intra-block edges
                    # p_neg = 0.2   # allow a few inter-block edges
                    p_neg = 0.1  # allow a few inter-block edges
                    ###Enes Son


                    sample = torch.where(
                        probs > p_pos, torch.ones_like(probs),
                        torch.where(probs < p_neg, torch.zeros_like(probs),
                                    torch.bernoulli(probs))
                    )
                    #sample = (probs > 0.28).float()           # threshold at 0.5 
                    #sample = torch.bernoulli(probs)
                    if i > 0:
                        # write predicted edges to all previous nodes
                        y_pred[0, i, :i] = sample[0, :i]

                 # 4) trim to the true size, then build adjacency matrix and graph
                # slice off only the first `length_b` rows, and up to max_prev
                y_trim = y_pred[0, :length_b, :].cpu().long().numpy()
                adj    = decode_adj(y_trim)
                # from_numpy_array keeps all nodes 0..maxN inclusive :contentReference[oaicite:5]{index=5}
                G   = get_graph(adj)             
                
                print(f"Graph {len(G_pred)+1:3d}: nodes={G.number_of_nodes()}, "
                      f"edges={G.number_of_edges()}")
                G_pred.append(G)

            if len(G_pred) >= total:
                break

    return G_pred

########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')

