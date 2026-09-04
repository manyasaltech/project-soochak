import numpy as np
import torch

# 8 ESP32 Node coordinates (Latitude, Longitude) around Panel 4-B
NODE_COORDS = np.array([
    [23.7501, 86.4120], # Node 1: Tx-1 Laser
    [23.7508, 86.4128], # Node 2: Rx-1 Detector
    [23.7515, 86.4135], # Node 3: MPU6050
    [23.7522, 86.4142], # Node 4: Load Cell
    [23.7529, 86.4149], # Node 5: MQ-4
    [23.7536, 86.4156], # Node 6: Tilt Sensor
    [23.7543, 86.4163], # Node 7: Boundary Sensor
    [23.7550, 86.4170]  # Node 8: Goaf Monitor
])

def build_adjacency_matrix(coords, sigma=0.002):
    """Calculates Gaussian distance-based spatial connections between nodes."""
    num_nodes = len(coords)
    adj = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                # Exponential distance decay for spatial correlation
                adj[i, j] = np.exp(-(dist**2) / (sigma**2))
                
    # Normalize row-wise (Degree Normalization)
    row_sum = adj.sum(axis=1, keepdims=True)
    adj_normalized = np.divide(adj, row_sum, where=row_sum!=0)
    return torch.tensor(adj_normalized, dtype=torch.float32)

if __name__ == "__main__":
    A = build_adjacency_matrix(NODE_COORDS)
    print("Graph Adjacency Matrix (8x8) built successfully.")
