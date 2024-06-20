import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import *
from testing_dataset import *
import time

class ann(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_of_hidden_layers):
        super(ann, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        for i in range(1, num_of_hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def train(model, input_tensor, target_tensor, loss_function, optimizer):
    start = time.time()
    num_epochs = 10000
    final_error = 0
    for epoch in range(num_epochs):
        current_loss = 0.0
        optimizer.zero_grad()
        # # Forward pass
        outputs = model(input_tensor)
        loss = loss_function(outputs, target_tensor)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            final_error = loss.item()
            # print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + ", Loss: " + str((loss.item())))
    print("Training has completed")
    print("Number of epochs : " + str(num_epochs))
    print("Total time taken for training : " + str(time.time() - start) + " seconds")
    return final_error

input_size = 9
hidden_size = 5
output_size = 1
num_of_hidden_layers = 1

model = ann(input_size, hidden_size, output_size, num_of_hidden_layers)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_error = train(model, input_tensor, target_tensor, loss_function, optimizer)
testing_error = 0

# test_input = torch.tensor([[0, 0, 8, 30, 1, 1, 1, 2, 1], [1, 3, 6, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 1, 0, 1, 1, 0]]).float() 
# test_output = torch.tensor([[1.0], [0.32333333333333336], [0.6]])
# print(test_input)

with torch.no_grad():
    start = time.time()
    outputs = model(testing_input_tensor)
    print("Time taken for testing : " + str(time.time() - start) + " seconds")
    # for i in range(len(outputs)):
        # print(f'Predicted output = {outputs[i].item()}')
    loss = loss_function(outputs, testing_target_tensor)
    testing_error = loss.item()

print("Training Error : " + str(training_error))
print("testing Error : " + str(testing_error))