import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.layers(x)

# Skip specifying seed value so that weights are randomly generated at every execution.
# Setting the seed to a specific value will cause the model to always output the same weights and ultimately the same answer until the seed valus is changed.
# torch.manual_seed(1000000)

input_size = 9
hidden_size = 5
output_size = 1

model = MLP(input_size, hidden_size, output_size)

print(model)

# input_tensor = torch.randn(32, input_size)  # 32 is the batch size
# target_tensor = torch.randn(32, output_size)
# input_tensor = torch.tensor([[2, 0, 12, 0, 1, 1, 1, 3, 1], [5, 5, 16, 90, 1, 1, 1, 1, 1], [10, 15, 8, 0, 0, 0, 1, 3, 1], [7, 15, 8, 90, 1, 1, 1, 4, 1], [9, 15, 4, 0, 1, 1, 1, 0, 1], [9, 17, 12, 45, 1, 0, 0, 1, 1], [10, 15, 12, 90, 1, 0, 1, 4, 1], [6, 25, 16, 90, 0, 0, 0, 0, 1], [6, 12, 12, 135, 1, 0, 0, 4, 1], [2, 7, 16, 0, 1, 1, 0, 2, 0], [3, 3, 12, 45, 1, 1, 0, 0, 0], [12, 22, 8, 0, 0, 0, 0, 4, 0], [2, 25, 12, 90, 1, 1, 0, 5, 1], [12, 17, 16, 135, 1, 1, 1, 3, 0], [4, 12, 4, 135, 1, 1, 1, 2, 0], [14, 25, 16, 135, 0, 1, 1, 1, 1], [0, 10, 16, 135, 1, 1, 0, 0, 0], [3, 3, 16, 180, 0, 1, 1, 5, 0], [12, 7, 16, 45, 1, 0, 0, 1, 0], [1, 22, 12, 90, 0, 1, 0, 2, 0], [12, 22, 8, 135, 0, 1, 0, 2, 0], [2, 5, 16, 180, 0, 0, 1, 2, 0], [13, 3, 0, 135, 1, 0, 0, 1, 0], [7, 0, 0, 90, 0, 0, 1, 1, 0], [2, 15, 8, 90, 0, 0, 0, 3, 0], [0, 15, 8, 180, 1, 0, 0, 2, 0], [0, 17, 12, 90, 1, 0, 1, 3, 0], [14, 25, 4, 135, 0, 0, 0, 4, 1], [6, 3, 12, 0, 1, 1, 1, 3, 1], [0, 7, 0, 135, 1, 0, 0, 2, 1], [11, 22, 8, 45, 1, 0, 0, 4, 1], [8, 25, 0, 180, 1, 1, 0, 5, 1], [6, 5, 4, 180, 1, 0, 0, 2, 0], [10, 3, 16, 0, 0, 1, 0, 0, 1], [13, 5, 4, 0, 0, 0, 0, 0, 1], [14, 17, 16, 90, 1, 0, 1, 1, 1], [5, 3, 4, 0, 1, 1, 0, 4, 0], [15, 17, 8, 45, 1, 1, 1, 0, 0], [11, 0, 8, 0, 1, 1, 1, 1, 1], [6, 3, 16, 45, 1, 1, 0, 4, 0], [0, 12, 12, 180, 1, 1, 0, 0, 0], [12, 15, 16, 135, 1, 0, 1, 4, 0], [11, 0, 16, 0, 0, 1, 1, 4, 0], [3, 5, 16, 180, 1, 0, 1, 5, 0], [0, 20, 8, 90, 0, 0, 0, 2, 0], [3, 10, 8, 180, 0, 0, 1, 4, 0], [8, 25, 12, 180, 0, 0, 0, 5, 0], [7, 10, 16, 45, 0, 1, 1, 1, 0], [9, 15, 16, 180, 0, 1, 1, 5, 0], [13, 10, 8, 90, 0, 1, 1, 1, 0]]).float()
# target_tensor = torch.tensor([[0.9833333333333332], [0.8133333333333332], [0.5766666666666665], [0.6933333333333334], [0.6266666666666666], [0.5599999999999999], [0.5766666666666665], [0.39666666666666667], [0.5866666666666667], [0.4966666666666667], [0.49000000000000005], [0.18], [0.6133333333333334], [0.37666666666666665], [0.4366666666666667], [0.6133333333333334], [0.35333333333333333], [0.5566666666666666], [0.36333333333333334], [0.2800000000000001], [0.24666666666666667], [0.3966666666666666], [0.35666666666666674], [0.4666666666666667], [0.21000000000000002], [0.26], [0.32666666666666666], [0.4133333333333334], [0.9066666666666666], [0.6466666666666666], [0.5133333333333334], [0.5633333333333334], [0.3133333333333333], [0.7233333333333334], [0.53], [0.5766666666666665], [0.5066666666666666], [0.3433333333333334], [0.95], [0.54], [0.35333333333333333], [0.27666666666666667], [0.6166666666666666], [0.43], [0.19666666666666666], [0.33666666666666667], [0.16333333333333333], [0.4533333333333333], [0.36000000000000004], [0.4033333333333333]]).float()
input_tensor = torch.tensor([[8, 10, 16, 45, 0, 0, 0, 3, 1], [10, 0, 8, 180, 1, 1, 1, 3, 0], [0, 5, 0, 180, 0, 0, 1, 5, 0], [14, 12, 8, 0, 1, 1, 0, 2, 0], [4, 5, 8, 0, 1, 0, 0, 3, 1], [5, 25, 4, 135, 1, 0, 1, 5, 0], [8, 7, 0, 135, 0, 1, 0, 0, 0], [8, 17, 12, 45, 1, 0, 1, 1, 1], [5, 17, 16, 0, 0, 0, 0, 2, 1], [1, 12, 12, 180, 0, 0, 0, 1, 0], [5, 12, 8, 0, 0, 0, 0, 5, 0], [1, 7, 8, 135, 0, 0, 1, 0, 0], [14, 12, 0, 45, 1, 1, 1, 3, 1], [10, 5, 16, 90, 0, 0, 1, 3, 0], [10, 12, 16, 90, 0, 0, 0, 2, 1], [5, 10, 8, 180, 0, 1, 0, 0, 0], [10, 0, 16, 135, 0, 1, 1, 2, 1], [13, 7, 8, 90, 0, 1, 0, 1, 1], [14, 15, 4, 90, 1, 1, 1, 0, 1], [14, 0, 8, 0, 1, 0, 0, 1, 0], [7, 3, 16, 0, 0, 1, 1, 5, 1], [12, 7, 4, 0, 1, 0, 0, 2, 1], [15, 15, 0, 45, 1, 1, 0, 2, 0], [1, 0, 8, 45, 1, 1, 0, 0, 1], [3, 0, 16, 0, 1, 0, 0, 5, 0], [2, 7, 12, 90, 1, 0, 1, 4, 1], [13, 5, 0, 45, 1, 1, 1, 3, 0], [11, 12, 8, 135, 1, 0, 0, 0, 1], [1, 10, 8, 90, 1, 0, 0, 1, 0], [9, 5, 4, 0, 1, 0, 0, 5, 0], [3, 25, 16, 135, 0, 0, 0, 5, 1], [0, 17, 4, 90, 1, 1, 0, 3, 1], [2, 5, 8, 90, 1, 1, 1, 2, 1], [12, 10, 4, 90, 0, 1, 1, 0, 1], [7, 7, 0, 45, 1, 1, 0, 4, 1], [7, 15, 8, 45, 0, 0, 1, 0, 1], [13, 17, 12, 90, 0, 1, 0, 4, 1], [8, 12, 8, 180, 0, 1, 0, 0, 1], [2, 10, 16, 90, 1, 0, 1, 4, 0], [3, 5, 16, 45, 0, 1, 1, 4, 0], [7, 25, 4, 0, 0, 0, 0, 1, 0], [13, 22, 12, 135, 0, 1, 1, 3, 1], [6, 3, 12, 45, 0, 0, 1, 5, 1], [2, 25, 16, 135, 0, 0, 1, 4, 0], [1, 12, 8, 0, 1, 0, 1, 0, 0], [15, 10, 4, 0, 1, 0, 0, 0, 0], [12, 15, 12, 45, 1, 1, 1, 3, 0], [13, 17, 12, 180, 1, 0, 1, 2, 0], [0, 10, 0, 0, 0, 0, 0, 0, 1], [1, 0, 4, 45, 1, 1, 0, 0, 0], [0, 3, 12, 45, 1, 0, 0, 2, 1], [1, 3, 12, 45, 1, 0, 0, 5, 0], [0, 0, 16, 0, 1, 1, 0, 2, 1], [1, 3, 12, 0, 0, 1, 0, 4, 0], [1, 0, 12, 0, 0, 1, 1, 2, 0], [1, 0, 16, 0, 1, 0, 0, 4, 0], [1, 3, 16, 0, 0, 0, 0, 1, 0], [1, 0, 12, 0, 1, 0, 0, 1, 1], [1, 0, 12, 0, 0, 0, 1, 3, 1], [0, 0, 12, 45, 1, 0, 1, 1, 0], [1, 3, 12, 45, 1, 1, 0, 3, 1], [0, 3, 16, 45, 0, 0, 1, 2, 1], [1, 0, 12, 45, 0, 0, 0, 1, 0], [0, 3, 16, 0, 0, 1, 0, 4, 0], [1, 0, 16, 0, 0, 1, 1, 1, 0], [0, 0, 16, 0, 1, 1, 0, 5, 1], [0, 0, 16, 0, 0, 0, 0, 5, 0], [0, 3, 16, 45, 1, 0, 1, 3, 0], [0, 0, 12, 45, 0, 0, 1, 4, 1], [1, 3, 16, 45, 0, 0, 0, 2, 0], [1, 3, 12, 0, 0, 0, 1, 1, 0], [0, 3, 12, 45, 1, 0, 1, 1, 1], [1, 3, 16, 45, 1, 0, 0, 5, 0], [0, 0, 12, 0, 0, 0, 0, 1, 1], [0, 3, 12, 0, 0, 0, 0, 3, 1], [1, 0, 16, 0, 1, 1, 1, 1, 1], [1, 0, 16, 0, 0, 1, 0, 3, 1], [1, 0, 12, 45, 1, 0, 0, 3, 0], [0, 0, 16, 45, 1, 0, 0, 1, 0], [0, 0, 12, 0, 1, 1, 1, 3, 0], [1, 0, 12, 45, 0, 1, 1, 5, 0], [1, 0, 16, 45, 1, 1, 1, 1, 1], [1, 0, 16, 0, 0, 1, 1, 2, 0], [1, 0, 12, 45, 0, 0, 1, 4, 0], [0, 0, 16, 45, 0, 0, 0, 3, 1], [0, 3, 16, 45, 0, 0, 0, 5, 1], [1, 0, 12, 45, 0, 0, 1, 2, 1], [1, 3, 16, 45, 1, 0, 1, 4, 1], [0, 3, 16, 0, 0, 0, 1, 4, 0], [0, 3, 12, 45, 0, 0, 1, 4, 0], [1, 3, 12, 45, 0, 0, 1, 5, 0], [1, 3, 16, 0, 1, 1, 1, 3, 0], [1, 0, 16, 0, 1, 1, 1, 1, 1], [0, 0, 12, 45, 0, 1, 0, 1, 0], [1, 0, 12, 0, 0, 1, 0, 1, 0], [1, 3, 16, 45, 1, 0, 0, 2, 0], [1, 3, 12, 45, 1, 0, 0, 5, 0], [1, 3, 12, 45, 1, 0, 0, 2, 0], [1, 0, 16, 0, 0, 1, 0, 2, 0], [1, 0, 12, 45, 0, 1, 1, 5, 0], [12, 23, 0, 180, 1, 1, 0, 0, 1], [11, 21, 0, 180, 0, 0, 0, 0, 0], [11, 21, 4, 180, 1, 0, 1, 0, 0], [11, 23, 4, 180, 1, 1, 1, 0, 0], [12, 23, 4, 180, 0, 1, 0, 0, 0], [13, 22, 4, 135, 1, 1, 0, 0, 0], [12, 22, 4, 135, 1, 0, 1, 0, 1], [13, 23, 4, 180, 0, 0, 1, 0, 0], [12, 22, 4, 180, 1, 1, 1, 0, 1], [11, 22, 4, 180, 1, 1, 0, 0, 0], [11, 23, 0, 180, 0, 0, 0, 0, 0], [13, 22, 0, 135, 1, 0, 0, 0, 1], [11, 21, 4, 180, 1, 0, 1, 0, 0], [13, 22, 4, 180, 0, 0, 0, 0, 1], [13, 23, 4, 180, 0, 1, 1, 0, 1], [12, 23, 4, 180, 1, 1, 0, 0, 0], [11, 22, 4, 135, 0, 0, 0, 0, 0], [13, 22, 4, 180, 1, 0, 1, 0, 0], [13, 23, 0, 180, 0, 0, 1, 0, 0], [12, 21, 4, 180, 1, 0, 0, 0, 1], [13, 22, 0, 135, 1, 0, 0, 0, 0], [12, 23, 0, 180, 1, 1, 1, 0, 1], [11, 23, 0, 135, 1, 1, 1, 0, 0], [13, 22, 0, 135, 0, 0, 1, 0, 0], [13, 21, 4, 180, 1, 0, 0, 0, 0], [11, 21, 0, 180, 1, 0, 1, 0, 0], [13, 23, 0, 135, 0, 0, 0, 0, 1], [11, 22, 4, 180, 0, 1, 1, 0, 1], [11, 21, 4, 180, 0, 1, 0, 0, 0], [11, 23, 4, 180, 0, 1, 1, 0, 1], [12, 22, 0, 135, 0, 1, 1, 0, 0], [11, 22, 0, 135, 1, 0, 0, 0, 0], [13, 23, 4, 135, 1, 1, 0, 0, 0], [13, 21, 0, 135, 0, 0, 1, 0, 0], [13, 21, 4, 135, 0, 0, 0, 0, 0], [12, 21, 0, 180, 0, 1, 1, 0, 1], [12, 22, 0, 180, 1, 0, 1, 0, 0], [12, 21, 0, 180, 1, 0, 0, 0, 1], [11, 21, 0, 135, 0, 0, 1, 0, 0], [13, 22, 4, 180, 1, 0, 1, 0, 0], [13, 21, 4, 180, 0, 0, 0, 0, 0], [12, 22, 0, 135, 0, 1, 0, 0, 1], [13, 21, 0, 135, 0, 1, 0, 0, 0], [11, 23, 4, 135, 0, 1, 1, 0, 1], [11, 22, 4, 180, 0, 1, 1, 0, 1], [12, 23, 4, 135, 1, 1, 1, 0, 0], [12, 22, 0, 135, 0, 1, 1, 0, 0], [13, 22, 4, 135, 0, 0, 0, 0, 0], [13, 22, 0, 180, 1, 0, 1, 0, 0], [12, 23, 4, 180, 0, 0, 0, 0, 0]]).float()
target_tensor = torch.tensor([[0.5866666666666667], [0.6166666666666666], [0.38], [0.4033333333333333], [0.6966666666666667], [0.22999999999999998], [0.3133333333333333], [0.6266666666666667], [0.5266666666666666], [0.27], [0.2866666666666667], [0.32999999999999996], [0.7366666666666666], [0.3633333333333333], [0.5366666666666667], [0.2866666666666667], [0.8833333333333334], [0.6966666666666667], [0.5766666666666668], [0.4833333333333334], [0.8733333333333334], [0.63], [0.31], [0.85], [0.5166666666666667], [0.73], [0.4966666666666667], [0.5033333333333333], [0.30333333333333334], [0.3466666666666666], [0.48], [0.6266666666666666], [0.8300000000000001], [0.6033333333333334], [0.7466666666666666], [0.5266666666666666], [0.5766666666666668], [0.5866666666666667], [0.37], [0.5299999999999999], [0.16333333333333333], [0.6133333333333334], [0.7733333333333334], [0.24666666666666667], [0.3366666666666667], [0.2033333333333333], [0.41], [0.27666666666666667], [0.52], [0.5166666666666666], [0.7733333333333333], [0.4566666666666667], [0.9333333333333332], [0.5233333333333334], [0.65], [0.5166666666666667], [0.42333333333333345], [0.8166666666666668], [0.85], [0.6], [0.8566666666666667], [0.8066666666666666], [0.4833333333333334], [0.54], [0.65], [0.9333333333333332], [0.5], [0.54], [0.8666666666666666], [0.42333333333333345], [0.49000000000000005], [0.8400000000000001], [0.4566666666666667], [0.8], [0.74], [0.9833333333333332], [0.8833333333333334], [0.5166666666666667], [0.5333333333333333], [0.7], [0.65], [0.9833333333333332], [0.65], [0.55], [0.8], [0.74], [0.85], [0.8233333333333335], [0.5066666666666666], [0.5066666666666666], [0.49000000000000005], [0.6233333333333333], [0.9833333333333332], [0.6], [0.5833333333333334], [0.4566666666666667], [0.4566666666666667], [0.4566666666666667], [0.5833333333333334], [0.65], [0.48], [0.04666666666666667], [0.14666666666666667], [0.24666666666666667], [0.14666666666666667], [0.18], [0.44666666666666666], [0.11333333333333333], [0.5466666666666666], [0.18], [0.04666666666666667], [0.38], [0.14666666666666667], [0.3466666666666667], [0.5133333333333334], [0.18], [0.04666666666666667], [0.14666666666666667], [0.11333333333333333], [0.38], [0.08], [0.5466666666666666], [0.24666666666666667], [0.11333333333333333], [0.08], [0.14666666666666667], [0.3466666666666667], [0.5133333333333334], [0.14666666666666667], [0.5133333333333334], [0.21333333333333337], [0.08], [0.18], [0.11333333333333333], [0.04666666666666667], [0.5133333333333334], [0.14666666666666667], [0.38], [0.11333333333333333], [0.14666666666666667], [0.04666666666666667], [0.44666666666666666], [0.14666666666666667], [0.5133333333333334], [0.5133333333333334], [0.24666666666666667], [0.21333333333333337], [0.04666666666666667], [0.14666666666666667], [0.04666666666666667]]).float()
# input_tensor = torch.tensor()
# target_tensor = torch.tensor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10000
for epoch in range(num_epochs):
    # print(f'Starting Epoch {epoch+1}')
    current_loss = 0.0
    optimizer.zero_grad()
    # # Forward pass
    outputs = model(input_tensor)
    loss = criterion(outputs, target_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Training has completed")
print(model.layers[0].weight)

test_input = torch.tensor([[0, 0, 8, 0, 1, 0, 1, 1, 0]]).float()  # Example input tensor
print(test_input)

model.eval()
with torch.no_grad():
    predicted_output = model(test_input)
    print("Predicted output:", predicted_output.item())





# input_size = 9
# hidden_size = 20
# output_size = 1

# # input_data = np.array([[0, 0, 8, 30, 1, 1, 1, 2, 1], [1, 3, 6, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 1, 0, 1, 1, 0]])
# # target = np.array([[1.0], [0.32333333333333336], [0.6]])

# input_data = np.array([[15, 22.5, 8, 180, 0, 1, 1, 5, 0], [11, 15, 16, 180, 0, 1, 0, 2, 1], [11, 12.5, 12, 180, 0, 0, 1, 3, 1], [0, 0, 0, 180, 1, 0, 0, 0, 0], [4, 0, 12, 180, 0, 0, 0, 3, 0], [9, 15, 4, 180, 1, 1, 0, 0, 1], [12, 10, 4, 90, 0, 0, 0, 4, 1], [15, 2.5, 16, 45, 1, 1, 1, 5, 1], [13, 25, 0, 180, 1, 0, 1, 3, 0], [0, 0, 4, 45, 0, 1, 1, 5, 0], [10, 20, 4, 180, 0, 0, 0, 0, 0], [5, 17.5, 0, 90, 1, 1, 1, 5, 0], [3, 25, 16, 0, 0, 0, 0, 0, 0], [8, 7.5, 0, 90, 1, 1, 0, 3, 1], [14, 17.5, 8, 45, 1, 0, 1, 0, 1], [0, 22.5, 16, 45, 0, 1, 0, 0, 1], [5, 17.5, 8, 90, 1, 1, 1, 2, 0], [2, 15, 12, 0, 1, 1, 0, 5, 0], [0, 2.5, 4, 0, 1, 1, 1, 2, 1], [12, 25, 4, 180, 0, 1, 1, 3, 1], [11, 17.5, 12, 0, 0, 0, 1, 4, 0], [10, 5, 0, 90, 1, 0, 0, 5, 0], [3, 20, 16, 180, 1, 0, 1, 0, 0], [10, 2.5, 0, 45, 0, 1, 0, 2, 0], [4, 17.5, 12, 180, 1, 0, 0, 3, 1], [3, 17.5, 8, 180, 0, 0, 0, 0, 0], [10, 2.5, 0, 180, 1, 1, 0, 4, 1], [12, 20, 0, 135, 0, 0, 1, 5, 1], [15, 15, 8, 135, 1, 1, 1, 1, 0], [7, 0, 4, 0, 1, 1, 1, 4, 0], [13, 20, 16, 0, 1, 1, 0, 2, 0], [15, 25, 4, 45, 0, 0, 1, 1, 1], [2, 5, 8, 45, 0, 0, 0, 3, 0], [5, 25, 12, 90, 0, 1, 0, 1, 1], [14, 25, 4, 135, 0, 0, 1, 4, 0], [0, 12.5, 4, 0, 0, 1, 0, 2, 1], [3, 2.5, 8, 0, 1, 0, 1, 4, 1], [12, 5, 16, 0, 1, 0, 0, 0, 0], [12, 0, 16, 90, 1, 1, 0, 0, 1], [7, 17.5, 12, 90, 0, 1, 1, 4, 0], [8, 25, 16, 0, 0, 1, 0, 4, 1], [15, 12.5, 8, 135, 0, 0, 0, 2, 0], [12, 12.5, 8, 0, 0, 1, 0, 3, 1], [3, 17.5, 4, 135, 0, 1, 0, 0, 0], [1, 5, 4, 90, 1, 1, 0, 5, 0], [1, 20, 0, 180, 1, 1, 1, 2, 0], [3, 20, 16, 45, 1, 0, 0, 5, 0], [3, 25, 12, 90, 1, 1, 0, 2, 0], [0, 7.5, 16, 180, 0, 1, 1, 2, 0], [13, 22.5, 4, 180, 0, 0, 1, 5, 0]])
# target = np.array([[0.31333333333333335], [0.5766666666666668], [0.6033333333333333], [0.4], [0.45], [0.5266666666666666], [0.5033333333333333], [0.89], [0.21333333333333332], [0.6333333333333334], [0.04666666666666667], [0.36000000000000004], [0.14666666666666667], [0.7133333333333334], [0.5433333333333333], [0.5633333333333332], [0.39333333333333337], [0.3766666666666667], [0.9066666666666666], [0.58], [0.27666666666666667], [0.2966666666666667], [0.21333333333333337], [0.4566666666666667], [0.5433333333333333], [0.14333333333333334], [0.7566666666666667], [0.48], [0.37666666666666665], [0.6333333333333334], [0.31333333333333335], [0.5133333333333333], [0.36333333333333334], [0.5633333333333334], [0.18], [0.6866666666666666], [0.8233333333333335], [0.2966666666666667], [0.7833333333333334], [0.36000000000000004], [0.5966666666666666], [0.23666666666666672], [0.67], [0.21000000000000002], [0.43000000000000005], [0.3466666666666667], [0.24666666666666667], [0.31333333333333335], [0.5133333333333333], [0.18]])

# # instatiate the MLP
# model = MLP(input_size, hidden_size, output_size)
# model.train(input_data, target)

# print("Trained")

# test_in = np.array([[0, 43, 0, 42, 0, 0, 0, 6, 0]])
# test_out = np.array([[0.19666666666666666]])
# output = model.forward(test_in)

# print(output)