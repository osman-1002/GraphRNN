import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphRNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(GraphRNNDecoder, self).__init__()

        # Graph-level RNN (to generate nodes)
        self.graph_rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Edge-level RNN (to generate edges)
        self.edge_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Output layers
        self.node_fc = nn.Linear(hidden_dim, output_dim)  # Predict next node
        self.edge_fc = nn.Linear(hidden_dim, 1)  # Predict edges (binary output)

    def forward(self, graph_embedding, max_nodes=10):
        """
        :param graph_embedding: Encoded graph vector from GNN encoder
        :param max_nodes: Maximum number of nodes to generate
        :return: Generated adjacency matrices
        """

        batch_size = graph_embedding.shape[0]
        h_graph = torch.zeros(4, batch_size, graph_embedding.shape[-1]).to(graph_embedding.device)
        h_edge = torch.zeros(4, batch_size, graph_embedding.shape[-1]).to(graph_embedding.device)

        generated_nodes = []
        generated_edges = []

        # Generate nodes sequentially
        node_input = graph_embedding.unsqueeze(1)  # Start with encoded graph
        for _ in range(max_nodes):
            h_graph, _ = self.graph_rnn(node_input, h_graph)
            node_prob = torch.sigmoid(self.node_fc(h_graph[:, -1, :]))
            generated_nodes.append(node_prob)

            # Generate edges for this node
            edge_input = h_graph[:, -1, :].unsqueeze(1)
            edge_predictions = []
            for _ in range(len(generated_nodes)):  # Connect to previous nodes
                h_edge, _ = self.edge_rnn(edge_input, h_edge)
                edge_prob = torch.sigmoid(self.edge_fc(h_edge[:, -1, :]))
                edge_predictions.append(edge_prob)

            generated_edges.append(torch.cat(edge_predictions, dim=-1))

        return torch.cat(generated_nodes, dim=-1), torch.stack(generated_edges)

# Example usage
if __name__ == "__main__":
    batch_size = 4
    encoder_output = torch.randn(batch_size, 128)  # Fake encoder output

    decoder = GraphRNNDecoder(input_dim=128, hidden_dim=64, output_dim=1)
    nodes, edges = decoder(encoder_output, max_nodes=5)

    print("Generated Nodes Shape:", nodes.shape)
    print("Generated Edges Shape:", edges.shape)
