from fastapi import APIRouter, HTTPException
import torch
import math
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
from pydantic import BaseModel
import os
import logging

router = APIRouter()

# Define the DCRNN model class
class DCRNN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_features):
        super(DCRNN, self).__init__()
        self.diff_conv1 = GCNConv(in_features, hidden_dim)
        self.diff_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.diff_conv1(x, edge_index))
        x = torch.relu(self.diff_conv2(x, edge_index))
        x = x.view(1, num_nodes, -1)  # Reshape for GRU
        x, _ = self.gru(x)
        x = self.fc(x)
        return x

# Junctions and connectivity
junctions = [
    {"junctionID": 2, "max": 100, "junction": "RVQV+X5W Lucknow", "Lat": 26.8400, "Long": 80.8929},
    {"junctionID": 3, "max": 100, "junction": "Kanpur Rd, Alambagh", "Lat": 26.8312, "Long": 80.9102},
    {"junctionID": 4, "max": 100, "junction": "RWMJ+9P6 Lucknow", "Lat": 26.8334, "Long": 80.9318},
    {"junctionID": 5, "max": 100, "junction": "Motilal Nehru Marg", "Lat": 26.8396, "Long": 80.9347},
    {"junctionID": 6, "max": 100, "junction": "Vidhan Sabha Marg", "Lat": 26.8429, "Long": 80.9420},
    {"junctionID": 7, "max": 100, "junction": "RWWW+JF8 Lucknow", "Lat": 26.8465, "Long": 80.9462},
    {"junctionID": 8, "max": 100, "junction": "VX42+5J7 Lucknow", "Lat": 26.8554, "Long": 80.9515},
]

# Define road connectivity
edges = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
node_map = {junction["junctionID"]: i for i, junction in enumerate(junctions)}
edges_mapped = [(node_map[a], node_map[b]) for a, b in edges]
edge_index = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()

# Model parameters
num_nodes = len(junctions)
num_features = 5  # Vehicle Count, Speed, Congestion, SinTime, CosTime
hidden_dim = 32
out_features = 3  # Predict Vehicle Count, Speed, Congestion

# Initialize the model
model = DCRNN(num_nodes, num_features, hidden_dim, out_features)

# Load the model weights
model.load_state_dict(torch.load("model/dcrnn_lucknow.pth", map_location=torch.device('cpu')))
model.eval()

# Helper functions
def encode_time(hour):
    """Encodes the hour of the day into sine and cosine components."""
    return math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)

def get_current_data(hour):
    """Returns a tensor of traffic features for each junction at the given hour."""
    sin_time, cos_time = encode_time(hour)
    return torch.tensor([
        [50, 40, 0.5, sin_time, cos_time],  # Junction 2
        [10, 35, 0.1, sin_time, cos_time],  # Junction 3
        [90, 30, 0.9, sin_time, cos_time],  # Junction 4
        [20, 45, 0.2, sin_time, cos_time],  # Junction 5
        [80, 38, 0.8, sin_time, cos_time],  # Junction 6
        [100, 25, 1.0, sin_time, cos_time], # Junction 7
        [55, 50, 0.55, sin_time, cos_time]  # Junction 8
    ], dtype=torch.float)



class PredictionRequest(BaseModel):
    hour: int

@router.post("/predict")
async def predict(request: PredictionRequest):
    
    hour = request.hour
  # Extract the hour from the request body

    # Get initial input data
    x = get_current_data(hour)
    graph_data = Data(x=x, edge_index=edge_index)

    # Prediction loop
    with torch.no_grad():
        # Predict for 10 minutes
        pred_10min = model(graph_data).squeeze(0)  # Expected shape: (7, 3)

        # Get time encoding
        sin_time, cos_time = encode_time(hour + 10 / 60)
        time_encoding = torch.tensor([[sin_time, cos_time]] * len(junctions), dtype=torch.float32)  # Shape: (7, 2)
        pred_10min = torch.cat([pred_10min, time_encoding], dim=-1)

        # Create new graph data for the next prediction
        pred_10min_data = Data(x=pred_10min, edge_index=edge_index)

        # Predict for 20 minutes
        pred_20min = model(pred_10min_data).squeeze(0)

        # Get time encoding
        sin_time, cos_time = encode_time(hour + 20 / 60)
        time_encoding = torch.tensor([[sin_time, cos_time]] * len(junctions), dtype=torch.float32)
        pred_20min = torch.cat([pred_20min, time_encoding], dim=-1)
        pred_20min_data = Data(x=pred_20min, edge_index=edge_index)

        # Predict for 30 minutes
        pred_30min = model(pred_20min_data).squeeze(0)

    # Prepare response
    return {
        "10_min": pred_10min[:, :3].tolist(),
        "20_min": pred_20min[:, :3].tolist(),
        "30_min": pred_30min[:, :3].tolist()
    }
