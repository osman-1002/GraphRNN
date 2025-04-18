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
                epoch, args.epochs,loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data*feature_dim
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


def test_rnn_dec_epoch(epoch, dataset_test, args, encoder, rnn, output, test_batch_size=16):
    # Set models to evaluation mode
    encoder.eval()
    rnn.eval()
    output.eval()
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    G_pred_list = []  # List to store generated graphs
    
    with torch.no_grad():  # Disable gradient computation for efficiency

        # Initialize the first batch to extract input dimensions for hidden state
        first_batch = next(iter(dataset_test))

        # Move batch data to GPU
        x = first_batch['x'].to(device)
        edge_index = first_batch['edge_index'].to(device)
        batch = first_batch['batch'].to(device)
        batch_size = dataset_test.batch_size  # Ensure correct batch size
        # Initialize hidden state once per epoch
        # Forward pass through encoder to get graph embeddings
        graph_embedding = encoder(x, edge_index, batch).float()
        
        # Normalize graph embedding
        #graph_embedding = (graph_embedding - graph_embedding.mean(dim=1, keepdim=True)) / (graph_embedding.std(dim=1, keepdim=True) + 1e-8)
        hidden = rnn.init_hidden(graph_embedding.repeat(1, batch_size, 1) + 0.1 * torch.randn_like(graph_embedding))
        # Add output projection layer to ensure output dimensions match target
        for batch_data in dataset_test:
            # Move batch data to GPU
            x = batch_data['x'].to(device)
            edge_index = batch_data['edge_index'].to(device)
            batch = batch_data['batch'].to(device)
            edge_seq = batch_data['edge_seq'].to(device)
            
           
            edge_seq = edge_seq.float()
            
            # Forward pass through encoder to get graph embeddings
            graph_embedding = encoder(x, edge_index, batch).float()
            
            
            # Ensure edge_seq has correct dtype and dimensions
            edge_seq = edge_seq.float()
            

            # Compute actual lengths of edge sequences
            edge_lengths = batch_data['len']
            sorted_indices = torch.argsort(edge_lengths, descending=True)
            edge_seq = edge_seq[sorted_indices]
            edge_lengths = edge_lengths[sorted_indices]

            
            # Pack the sequence before feeding into RNN
            packed_x = pack_padded_sequence(x, edge_lengths.cpu(), batch_first=True, enforce_sorted=False)
            # RNN forward pass
            output_seq, hidden = rnn(packed_x, graph_embedding.unsqueeze(0), hidden)
            if isinstance(output_seq, PackedSequence):
                # Unpack sequence
                output_seq, _ = pad_packed_sequence(output_seq, batch_first=True)

            output_seq = torch.sigmoid(output_seq)
            # Detach hidden state to prevent backprop through old computation graphs
            hidden = hidden.detach()
            for i in range(test_batch_size):
                print("Output sequence shape:", output_seq.shape)

                adj_output = torch.tanh(output_seq[i])  # Squashes values into [-1, 1] range
                adj_output = (adj_output + 1) / 2  # Rescale to [0,1]
                adj_output = adj_output.cpu().detach().numpy()  # Convert to NumPy

                # Decode adjacency matrix
                adj_pred = dec_decode_adj(adj_output)

                print("Adjacency Matrix Shape:", adj_pred.shape)
                G_pred = get_graph(adj_pred)  # Convert to a graph
                print("Generated Graph Info:", nx.info(G_pred))
                G_pred_list.append(G_pred)

    return G_pred_list


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
        print("Output sequence shape:", y_pred_long_data[i].shape)
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


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1


    # Ensure that the model and data are on the same device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move the model to the correct device
    rnn.to(device)
    output.to(device)
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        print(epoch)
        now = datetime.datetime.now()
        print(now.time())
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_ATT' in args.note:
            train_rnn_epoch_with_attention(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            print(epoch)
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    print(len(G_pred))
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    elif 'GraphRNN_ATT' in args.note:
                        G_pred_step = test_rnn_with_attention_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN'  in args.note:
                    break
                elif 'GraphRNN_ATT' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)

def train_dec(args, dataset_train, dataset_test, encoder, rnn, output):
    # Check if we need to load existing models
    if args.load:
        fname_rnn = args.model_save_path + args.fname + 'rnn_' + str(args.load_epoch) + '.dat'
        fname_output = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname_rnn))
        output.load_state_dict(torch.load(fname_output))
        args.lr = 0.00001
        epoch = args.load_epoch
        print(f'Model loaded! Continuing from epoch {epoch}, with learning rate {args.lr}')
    else:
        epoch = 1

    # Set device for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    rnn.to(device)
    output.to(device)

    # Initialize optimizers
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_rnn = optim.Adam(rnn.parameters(), lr=args.lr)
    optimizer_output = optim.Adam(output.parameters(), lr=args.lr)

    # Learning rate schedulers
    scheduler_encoder = MultiStepLR(optimizer_encoder, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # Track timing for each epoch
    time_all = np.zeros(args.epochs)

    while epoch <= args.epochs:
        print(f'\nEpoch {epoch}/{args.epochs}')
        epoch_start_time = tm.time()

        encoder.train()
        rnn.train()
        output.train()
        # Initialize the first batch to extract input dimensions for hidden state
        first_batch = next(iter(dataset_train))

        # Move batch data to GPU
        x = first_batch['x'].to(device)
        edge_index = first_batch['edge_index'].to(device)
        batch = first_batch['batch'].to(device)
        batch_size = dataset_train.batch_size  # Ensure correct batch size
        # Initialize hidden state once per epoch
        # Forward pass through encoder to get graph embeddings
        graph_embedding = encoder(x, edge_index, batch).float()
        
        hidden = rnn.init_hidden(graph_embedding.repeat(1, batch_size, 1) + 0.1 * torch.randn_like(graph_embedding))
                
        for batch_data in dataset_train:
            # Move batch data to GPU
            x = batch_data['x'].to(device)
            y = batch_data['y'].to(device)
            edge_index = batch_data['edge_index'].to(device)
            batch = batch_data['batch'].to(device)
            edge_seq = batch_data['edge_seq'].to(device)
            
            edge_seq = edge_seq.float()
            
            # Forward pass through encoder to get graph embeddings
            graph_embedding = encoder(x, edge_index, batch).float()
            
            # Ensure edge_seq has correct dtype and dimensions
            edge_seq = edge_seq.float()
            

            # Compute actual lengths of edge sequences
            edge_lengths = batch_data['len']

            sorted_indices = torch.argsort(edge_lengths, descending=True)
            edge_seq = edge_seq[sorted_indices]
            edge_lengths = edge_lengths[sorted_indices]
            # Pack the sequence before feeding into RNN
            packed_x = pack_padded_sequence(x, edge_lengths.cpu(), batch_first=True, enforce_sorted=False)
            # RNN forward pass
            output_seq, hidden = rnn(packed_x, graph_embedding.unsqueeze(0), hidden)
            if isinstance(output_seq, PackedSequence):
                # Unpack sequence
                output_seq, _ = pad_packed_sequence(output_seq, batch_first=True)

            output_seq = torch.sigmoid(output_seq)
            target_seq = y
            bce_loss = binary_cross_entropy_weight(output_seq, target_seq)

            # Compute community loss from predicted adjacency
            adj_output = output_seq[0].cpu().detach().numpy()
            adj_pred = dec_decode_adj(adj_output)
            comm_loss = approximate_community_loss(adj_pred)

            λ = 0.1  # Regularization strength
            loss = bce_loss + λ * comm_loss            # Backpropagation
            optimizer_encoder.zero_grad()
            optimizer_rnn.zero_grad()
            optimizer_output.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(output.parameters(), max_norm=1.0)
            
            optimizer_encoder.step()
            optimizer_rnn.step()
            optimizer_output.step()
            
            # Detach hidden state to prevent backprop through old computation graphs
            hidden = hidden.detach()
    
        # Update learning rate
        scheduler_encoder.step()
        scheduler_rnn.step()
        scheduler_output.step()
        
        # Log epoch information
        epoch_time = tm.time() - epoch_start_time
        time_all[epoch - 1] = epoch_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds. Loss: {loss.item():.4f}')
        
        # Save model checkpoints
        if args.save and epoch % args.epochs_save == 0:
            torch.save(encoder.state_dict(), f"{args.model_save_path}{args.fname}encoder_{epoch}.dat")
            torch.save(rnn.state_dict(), f"{args.model_save_path}{args.fname}rnn_{epoch}.dat")
            torch.save(output.state_dict(), f"{args.model_save_path}{args.fname}output_{epoch}.dat")
            print(f'Models saved for epoch {epoch}.')
        
        epoch += 1


    # Save timing information
    np.save(args.timing_save_path + args.fname, time_all)
    print('Training completed.')
    
    epoch -= 1
    # test
    print(f"Epoch: {epoch}, epochs_test: {args.epochs_test}, epochs_test_start: {args.epochs_test_start}")
    if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
        print(f"Testing at epoch {epoch}...")
        for sample_time in range(1, 4):
            G_pred = []
            while len(G_pred) < args.test_total_size:
                print(f"Generating graph {len(G_pred)} / {args.test_total_size}")
                G_pred_step = test_rnn_dec_epoch(epoch, dataset_test, args, encoder, rnn, output, test_batch_size=args.test_batch_size)
                G_pred.extend(G_pred_step)
            # Save graphs
            # Save graphs after generating enough samples
            fname = os.path.join(args.graph_save_path, f"{args.fname_pred}{epoch}_{sample_time}.dat")
            with open(fname, 'wb') as f:
                pickle.dump(G_pred, f)
            print(f"Graphs saved at {fname}")
    print('Test done.')
    
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

