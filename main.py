from train import *
from encoder import GraphEncoder

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y%m%d-%H%M%S")
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    graphs = create_graphs.create(args)
    
    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_validate = graphs[0:int(0.2*graphs_len)]

    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]


    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)



    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
        # for edge in list(graph.edges()):
        #     # print('edge',edge)
        #     if np.random.rand()>p:
        #         graph.remove_edge(edge[0],edge[1])


    # ## dataset initialization
    # if 'nobfs' in args.note:
    #     print('nobfs')
    #     dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
    #     args.max_prev_node = args.max_num_node-1
    # if 'barabasi_noise' in args.graph_type:
    #     print('barabasi_noise')
    #     dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
    #     args.max_prev_node = args.max_num_node - 1
    # else:
    #     dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
    #                                                                  num_samples=args.batch_size*args.batch_ratio, replacement=True)
    # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                                            sampler=sample_strategy)
    ## Prepare dataset loader
    if 'nobfs' in args.note:
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
    elif 'barabasi_noise' in args.graph_type:
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train, max_prev_node=args.max_prev_node)
    else:
        dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node)


    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy, collate_fn=custom_collate)
    
    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()
    for batch_data in dataset_loader:
        for key, value in batch_data.items():
            print(f"{key}: shape {value.shape}")
        break  # Only need to check once
    # Initialize models
    if 'GraphRNN_ATT' in args.note:
        # Encoder: input_dim = max_prev_node, hidden_dim = hidden_size_rnn,
        #           attention_dim = embedding_size_rnn, output_dim = hidden_size_rnn_output
        # encoder = GRUWithAttention(
        #     input_size=args.max_prev_node,
        #     hidden_size=args.hidden_size_rnn,
        #     attn_size=args.embedding_size_rnn,      # dimension of the alignment network
        #     output_size=args.hidden_size_rnn_output,
        #     num_layers=args.num_layers
        # ).cuda()
        
        encoder=GRU_plain(input_size=args.max_prev_node, 
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn, 
            num_layers=args.num_layers, 
            has_input=True,
            has_output=True, 
            output_size=args.hidden_size_rnn_output
        ).cuda()

        # Decoder: input_dim = 1, hidden_dim = hidden_size_rnn_output,
        #           attention_dim = embedding_size_rnn_output, output_dim = 1
        decoder = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()
    else:
        encoder = GraphEncoder(input_dim=80, hidden_dim=args.hidden_size_rnn, 
                            output_dim=args.hidden_size_rnn, num_layers=args.num_layers, 
                                use_attention=False).cuda()
        decoder = GRU_flat_dec_multihead(input_size=2, 
                                    embedding_size=args.embedding_size_rnn,
                                    hidden_size=args.hidden_size_rnn,
                                    num_layers=args.num_layers,
                                    edge_output_size=args.max_prev_node,
                                    value_output_size=80).cuda()
   
    # # compute empirical p_data once (or just set p_data=0.17)
    # p_data = 0.17
    # import math
    # logit_p = math.log(p_data/(1-p_data))
    # # the last layer of your edge‑head MLP is decoder.output_edge[-1]
    # decoder.output_edge[-1].bias.data.fill_(logit_p)
    # print("edge‑head bias initialized so sigmoid(bias)=%.2f" % p_data)
    # rnn = GRU_plain_dec(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
    #                     hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
    #                     has_output=True, output_size=80).cuda()
    # output = GRU_plain_dec(input_size=1, embedding_size=args.embedding_size_rnn_output,
    #                     hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
    #                     has_output=True, output_size=1).cuda()    

    ### start training

    train(args, dataset_loader, encoder, decoder)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

    # elif 'GraphRNN_RNN' in args.note:
    #     rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
    #                     hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
    #                     has_output=True, output_size=args.hidden_size_rnn_output).cuda()
    #     output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
    #                        hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
    #                        has_output=True, output_size=1).cuda()