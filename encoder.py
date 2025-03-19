import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_attention=False):
        super(GraphEncoder, self).__init__()
        self.use_attention = use_attention

        if use_attention:
            self.convs = nn.ModuleList([GATConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        else:
            self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

        self.fc = nn.Linear(hidden_dim, output_dim)  # Project to match GRU_plain hidden size

    def forward(self, x, edge_index, batch):
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]  # Filter valid edges

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        batch = batch.view(-1)          # Flatten batch to 1D
        x = x.view(-1, x.size(-1))      # Flatten x to match batch size

        graph_embedding = global_mean_pool(x, batch)
        return self.fc(graph_embedding)  # Ensure output matches GRU_plain hidden size

# Example usage
if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    dataset = TUDataset(root="data", name="PROTEINS")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = GraphEncoder(input_dim=dataset.num_features, hidden_dim=64, output_dim=128, use_attention=True)

    for batch in loader:
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        graph_embedding = encoder(x, edge_index, batch_idx)
        print("Graph Embedding Shape:", graph_embedding.shape)
        break  # Just to test one batch