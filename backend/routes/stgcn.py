from fastapi import APIRouter
import torch
from pydantic import BaseModel
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os

router = APIRouter()

# Define STGCN Model
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_features):
        super(STGCN, self).__init__()
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_features)
        self.gru = nn.GRU(out_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))
        x = x.unsqueeze(0)  # Add batch dimension for GRU
        x, _ = self.gru(x)
        x = self.fc(x)
        return x.squeeze(0)  # Remove batch dimension before returning

# Load the STGCN Model
def load_stgcn_model(model_path="model/stgcn_traffic.pth"):
    num_nodes = 7
    num_features = 3
    hidden_dim = 16
    out_features = 3

    model = STGCN(num_nodes, num_features, hidden_dim, out_features)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Ensure CPU compatibility
    model.eval()
    return model

model = load_stgcn_model()

# Define the Edge List and Convert to Tensor
edges = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
node_list = sorted(set(sum(edges, ())))  # Get unique nodes sorted
node_to_index = {j: i for i, j in enumerate(node_list)}
edge_index = torch.tensor(
    [[node_to_index[e[0]], node_to_index[e[1]]] for e in edges] +
    [[node_to_index[e[1]], node_to_index[e[0]]] for e in edges],  # Add reverse edges
    dtype=torch.long
).t().contiguous()

# Input Schema
class TrafficInput(BaseModel):
    data: list[list[float]]  # Expects [[50, 40, 0.5], ...]

@router.post("/predict")
async def predict_traffic(input_data: TrafficInput):
    test_data = torch.tensor(input_data.data, dtype=torch.float)

    if test_data.shape[0] != len(node_list):
        return {"error": "Input data does not match expected node count"}

    test_graph = Data(x=test_data, edge_index=edge_index)

    with torch.no_grad():
        pred_10min = model(test_graph).numpy()
        test_graph_20min = Data(x=torch.tensor(pred_10min, dtype=torch.float), edge_index=edge_index)
        pred_20min = model(test_graph_20min).numpy()

    return {"10_min": pred_10min.tolist(), "20_min": pred_20min.tolist()}
