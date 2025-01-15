import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import open3d as o3d

# 1. Custom Dataset for Point Clouds
class LiDARPointCloudDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.file_paths[idx])
        points = np.asarray(pcd.points)
        intensities = np.asarray(pcd.colors)[:, 0]  # Assuming intensity in colors
        label = self.labels[idx]
        return points, intensities, label

# 2. Deep Neural Network (based on PointNet)
class PointNetLaneDetector(nn.Module):
    def __init__(self):
        super(PointNetLaneDetector, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)  # Input: x, y, z, intensity
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Output: lane marking prediction
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# 3. Training Pipeline
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for points, intensities, labels in dataloader:
            # Prepare data
            points = points.float().permute(0, 2, 1)  # [batch, 4, num_points]
            labels = labels.float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 4. Data Preparation
file_paths = ["path/to/point_cloud1.ply", "path/to/point_cloud2.ply"]
labels = [0.5, 0.3]  # Example labels: distance to lane center
dataset = LiDARPointCloudDataset(file_paths, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Model Initialization
model = PointNetLaneDetector()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train the Model
train_model(model, dataloader, criterion, optimizer, num_epochs=10)

# 7. Save the Model
torch.save(model.state_dict(), "lane_detector_model.pth")

# 8. Inference Example
def infer_lane_distance(model, point_cloud_path):
    model.eval()
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    intensities = np.asarray(pcd.colors)[:, 0]
    input_data = torch.tensor(np.hstack((points, intensities[:, None])), dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.item()

# Example inference
prediction = infer_lane_distance(model, "path/to/test_point_cloud.ply")
print(f"Predicted lane distance: {prediction:.2f} meters")