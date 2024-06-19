import torch

testing_input_tensor = torch.tensor([[3, 17, 0, 90, 0, 1, 1, 0, 0], [12, 12, 8, 90, 1, 0, 0, 3, 1], [4, 0, 12, 180, 0, 1, 0, 5, 1], [1, 20, 0, 180, 0, 1, 1, 4, 0], [13, 17, 16, 45, 0, 1, 0, 3, 0], [15, 10, 0, 135, 1, 1, 1, 5, 1], [4, 25, 12, 135, 0, 0, 0, 3, 0], [4, 20, 16, 45, 1, 0, 0, 4, 0], [10, 25, 4, 0, 0, 0, 1, 4, 0], [1, 15, 16, 135, 1, 0, 1, 3, 1], [10, 12, 4, 135, 1, 1, 1, 0, 0], [8, 15, 12, 45, 0, 0, 1, 0, 0], [9, 7, 12, 90, 1, 0, 1, 4, 1], [15, 17, 4, 0, 1, 1, 1, 0, 0], [13, 17, 0, 90, 1, 1, 0, 5, 1], [7, 22, 8, 135, 0, 0, 1, 1, 1], [1, 20, 12, 45, 0, 1, 1, 0, 1], [12, 3, 4, 135, 1, 1, 1, 1, 0], [10, 10, 12, 90, 1, 0, 0, 4, 1], [12, 3, 8, 0, 1, 1, 1, 4, 0], [14, 20, 12, 45, 0, 1, 1, 5, 1], [3, 25, 4, 0, 1, 1, 1, 5, 1], [7, 20, 0, 90, 1, 0, 1, 5, 1], [10, 20, 4, 0, 0, 0, 1, 5, 0], [3, 25, 12, 0, 1, 0, 0, 5, 0], [14, 5, 8, 135, 0, 1, 1, 4, 0], [12, 7, 4, 135, 0, 0, 1, 5, 0], [15, 7, 0, 0, 0, 1, 1, 0, 0], [11, 12, 0, 135, 0, 0, 0, 1, 0], [14, 25, 12, 90, 1, 0, 0, 3, 1], [7, 10, 8, 45, 0, 0, 0, 2, 0], [1, 0, 4, 0, 1, 0, 1, 3, 1], [2, 12, 12, 0, 0, 0, 1, 4, 1], [14, 25, 0, 0, 0, 0, 0, 5, 1], [9, 7, 8, 90, 1, 1, 0, 1, 0], [2, 20, 16, 0, 1, 1, 0, 5, 1], [15, 5, 16, 180, 1, 0, 0, 2, 1], [10, 12, 16, 45, 1, 1, 1, 2, 1], [1, 25, 4, 135, 1, 1, 1, 0, 0], [10, 10, 0, 180, 1, 1, 1, 2, 0], [7, 12, 0, 45, 1, 1, 1, 1, 0], [9, 7, 16, 0, 1, 0, 1, 5, 0], [9, 22, 16, 90, 1, 1, 1, 4, 1], [7, 0, 8, 180, 0, 0, 0, 0, 1], [11, 15, 0, 90, 1, 0, 1, 1, 1], [6, 15, 4, 0, 1, 0, 1, 4, 0], [3, 0, 4, 90, 0, 1, 0, 3, 1], [7, 20, 4, 135, 0, 1, 1, 0, 1], [6, 20, 4, 90, 1, 0, 1, 5, 0], [14, 17, 8, 135, 1, 1, 1, 1, 1], [14, 22, 4, 90, 0, 1, 1, 3, 1], [2, 3, 8, 0, 1, 0, 0, 1, 1], [12, 3, 4, 135, 0, 1, 0, 1, 0], [0, 17, 12, 90, 0, 1, 1, 3, 1], [2, 20, 8, 0, 0, 0, 0, 4, 1], [9, 15, 8, 180, 1, 0, 0, 2, 0], [4, 17, 0, 90, 0, 1, 1, 1, 0], [12, 0, 16, 180, 0, 0, 1, 2, 0], [1, 0, 12, 135, 1, 0, 0, 5, 1], [6, 12, 0, 0, 1, 1, 1, 0, 0], [0, 20, 16, 180, 0, 1, 1, 3, 0], [0, 3, 12, 180, 1, 1, 1, 1, 1], [13, 5, 12, 0, 0, 0, 0, 1, 1], [7, 5, 16, 135, 0, 0, 1, 4, 0], [1, 22, 4, 0, 0, 1, 1, 0, 1], [9, 10, 12, 135, 0, 0, 0, 5, 0], [11, 3, 0, 0, 1, 0, 1, 4, 1], [10, 20, 4, 180, 1, 0, 1, 1, 0], [11, 20, 8, 0, 1, 0, 1, 3, 1], [9, 25, 8, 90, 1, 1, 0, 1, 0], [13, 25, 8, 45, 1, 0, 0, 2, 1], [9, 25, 16, 180, 0, 1, 1, 3, 1], [14, 0, 0, 180, 0, 0, 0, 1, 0], [6, 5, 0, 0, 0, 1, 0, 1, 1], [12, 15, 12, 90, 0, 1, 1, 4, 1], [14, 10, 16, 0, 0, 1, 0, 5, 0], [4, 7, 8, 45, 0, 1, 0, 3, 1], [0, 0, 12, 135, 0, 0, 0, 1, 0], [9, 20, 8, 45, 1, 1, 1, 1, 0], [13, 22, 8, 0, 1, 1, 1, 5, 0], [9, 10, 8, 180, 0, 0, 0, 2, 1], [8, 5, 4, 135, 0, 0, 1, 3, 1], [7, 5, 0, 135, 0, 0, 1, 4, 0], [6, 3, 16, 180, 0, 1, 0, 1, 0], [6, 17, 0, 0, 0, 1, 1, 4, 0], [6, 22, 12, 180, 1, 0, 0, 0, 1], [0, 17, 12, 90, 0, 0, 1, 1, 1], [0, 15, 12, 0, 1, 0, 1, 5, 0], [6, 0, 12, 45, 0, 0, 1, 1, 0], [14, 0, 8, 180, 0, 0, 0, 0, 1], [0, 5, 12, 180, 1, 0, 1, 2, 0], [3, 12, 8, 45, 0, 1, 0, 5, 0], [5, 17, 16, 90, 1, 1, 0, 3, 0], [15, 12, 12, 135, 0, 1, 1, 0, 1], [10, 10, 16, 45, 1, 0, 0, 2, 0], [7, 3, 16, 180, 1, 1, 0, 5, 1], [15, 22, 12, 180, 1, 1, 1, 0, 1], [1, 12, 16, 0, 0, 0, 0, 0, 0], [13, 17, 0, 45, 0, 1, 0, 1, 1], [10, 25, 12, 180, 0, 1, 1, 5, 1], [2, 12, 0, 0, 1, 0, 0, 0, 1], [4, 5, 12, 90, 0, 1, 0, 1, 1], [11, 0, 12, 90, 1, 0, 1, 4, 0], [4, 10, 4, 90, 0, 0, 1, 4, 1], [0, 22, 0, 0, 0, 1, 1, 1, 0], [15, 10, 4, 45, 0, 0, 1, 4, 1], [0, 22, 16, 135, 0, 0, 0, 2, 1], [1, 22, 16, 90, 1, 0, 1, 0, 1], [7, 5, 8, 135, 0, 0, 0, 1, 1], [2, 17, 16, 135, 1, 0, 0, 0, 0], [1, 15, 4, 180, 1, 0, 0, 5, 0], [2, 12, 12, 0, 0, 1, 1, 0, 0], [14, 7, 12, 90, 1, 1, 0, 0, 0], [10, 3, 4, 45, 1, 0, 1, 1, 0], [8, 0, 16, 180, 1, 0, 1, 3, 1], [10, 20, 8, 0, 0, 1, 1, 1, 0], [13, 12, 12, 135, 1, 0, 0, 3, 0], [11, 25, 8, 45, 0, 0, 0, 2, 1], [13, 7, 0, 0, 0, 1, 1, 1, 1], [9, 10, 8, 180, 1, 1, 0, 0, 0], [7, 22, 8, 90, 0, 1, 0, 5, 0], [0, 0, 0, 90, 0, 1, 1, 5, 1], [7, 22, 0, 45, 0, 0, 1, 0, 1], [10, 12, 4, 90, 0, 0, 1, 4, 1], [8, 10, 12, 90, 1, 1, 1, 3, 1], [4, 20, 0, 90, 0, 0, 1, 1, 0], [2, 7, 8, 135, 1, 0, 0, 1, 0], [7, 3, 12, 45, 1, 1, 1, 0, 1], [7, 20, 4, 45, 1, 0, 1, 5, 1], [4, 22, 8, 0, 0, 0, 0, 1, 0], [5, 5, 16, 45, 1, 0, 1, 2, 0], [8, 3, 16, 90, 0, 0, 0, 4, 1], [12, 20, 0, 0, 0, 1, 1, 3, 1], [5, 17, 4, 45, 0, 1, 1, 4, 1], [9, 12, 8, 45, 0, 1, 1, 5, 0], [7, 7, 16, 135, 1, 1, 1, 4, 1], [15, 12, 0, 90, 1, 0, 1, 1, 1], [13, 3, 12, 90, 0, 1, 1, 0, 1], [2, 15, 0, 45, 0, 1, 0, 2, 1], [8, 15, 12, 45, 0, 1, 1, 1, 0], [14, 20, 4, 45, 0, 0, 1, 1, 0], [9, 7, 8, 0, 0, 1, 0, 3, 1], [9, 7, 8, 0, 1, 0, 1, 0, 0], [15, 22, 8, 180, 1, 1, 1, 5, 0], [8, 12, 0, 90, 0, 1, 1, 0, 0], [14, 20, 4, 45, 0, 1, 1, 3, 1], [12, 20, 8, 45, 1, 0, 0, 4, 0], [15, 20, 16, 90, 0, 0, 0, 1, 0], [11, 7, 12, 135, 1, 1, 0, 3, 0], [10, 15, 4, 135, 1, 0, 1, 5, 1], [1, 3, 12, 0, 1, 0, 1, 1, 1], [0, 0, 12, 0, 0, 1, 1, 5, 1], [0, 3, 12, 0, 0, 0, 0, 4, 0], [0, 3, 12, 0, 1, 0, 1, 5, 0], [0, 0, 12, 45, 1, 0, 0, 4, 0], [0, 3, 16, 0, 0, 1, 1, 2, 0], [1, 0, 12, 45, 0, 0, 1, 3, 0], [0, 3, 16, 0, 1, 1, 0, 4, 1], [0, 3, 12, 0, 0, 0, 1, 2, 1], [1, 3, 16, 0, 1, 1, 1, 1, 0], [1, 0, 16, 45, 1, 0, 1, 1, 0], [0, 3, 16, 0, 1, 1, 1, 1, 0], [0, 3, 16, 0, 1, 1, 0, 1, 1], [0, 3, 16, 0, 0, 1, 1, 1, 0], [0, 0, 16, 45, 0, 1, 0, 4, 0], [0, 3, 12, 45, 1, 0, 0, 5, 1], [0, 3, 16, 45, 1, 1, 1, 3, 0], [1, 3, 12, 45, 0, 0, 1, 5, 1], [1, 3, 12, 45, 1, 0, 0, 4, 0], [0, 3, 12, 45, 0, 1, 0, 5, 1], [0, 0, 16, 45, 1, 0, 1, 4, 0], [1, 0, 12, 0, 1, 0, 1, 2, 0], [1, 0, 16, 45, 0, 1, 1, 2, 0], [1, 0, 12, 45, 1, 0, 0, 1, 1], [0, 3, 16, 0, 0, 0, 0, 3, 1], [1, 0, 16, 45, 1, 1, 0, 4, 1], [0, 0, 16, 45, 1, 0, 1, 4, 1], [0, 3, 12, 45, 0, 0, 0, 1, 1], [1, 0, 16, 0, 1, 0, 1, 2, 0], [0, 3, 12, 0, 0, 1, 0, 5, 0], [1, 0, 16, 45, 1, 0, 0, 2, 0], [0, 0, 12, 0, 0, 1, 0, 1, 0], [0, 0, 12, 45, 0, 0, 1, 3, 1], [0, 0, 12, 45, 1, 1, 1, 2, 1], [0, 0, 16, 45, 0, 1, 0, 1, 1], [0, 3, 12, 45, 1, 0, 1, 1, 1], [1, 3, 16, 0, 0, 0, 0, 2, 0], [1, 0, 12, 45, 1, 1, 1, 2, 0], [1, 0, 12, 0, 0, 0, 0, 1, 1], [0, 3, 12, 0, 0, 1, 1, 5, 0], [0, 3, 12, 0, 0, 1, 1, 2, 0], [0, 0, 12, 0, 1, 0, 1, 2, 0], [0, 0, 16, 45, 0, 0, 0, 5, 0], [0, 3, 12, 0, 0, 0, 0, 5, 1], [0, 3, 12, 45, 1, 0, 0, 5, 1], [0, 3, 12, 0, 0, 0, 1, 1, 1], [1, 3, 16, 45, 0, 0, 0, 4, 1], [1, 0, 12, 45, 0, 0, 0, 1, 1], [1, 0, 12, 0, 1, 1, 1, 1, 1], [1, 3, 12, 45, 1, 1, 1, 3, 0], [1, 0, 12, 0, 0, 0, 1, 4, 1], [0, 0, 16, 0, 0, 1, 0, 2, 1], [1, 0, 12, 45, 0, 1, 1, 1, 1], [1, 0, 16, 0, 0, 0, 1, 5, 1], [1, 3, 12, 45, 1, 1, 1, 4, 1], [0, 3, 12, 0, 0, 0, 1, 2, 0], [1, 3, 16, 0, 0, 0, 0, 2, 1], [0, 0, 16, 0, 0, 0, 1, 2, 0], [1, 0, 16, 0, 0, 0, 0, 5, 1], [0, 0, 12, 45, 1, 0, 1, 1, 0], [0, 0, 12, 0, 1, 0, 0, 2, 1], [1, 0, 16, 0, 0, 1, 1, 2, 1], [1, 0, 12, 45, 0, 1, 1, 2, 1], [1, 3, 12, 0, 1, 1, 1, 5, 0], [1, 3, 12, 45, 1, 1, 1, 4, 1], [0, 0, 12, 45, 0, 0, 1, 1, 1], [1, 3, 12, 0, 0, 0, 0, 5, 1], [1, 3, 16, 0, 1, 1, 0, 3, 1], [1, 0, 12, 0, 0, 0, 0, 1, 0], [0, 3, 16, 0, 1, 0, 1, 5, 0], [0, 0, 12, 45, 1, 1, 1, 1, 1], [0, 3, 16, 0, 0, 0, 0, 3, 1], [1, 3, 12, 0, 0, 0, 1, 3, 0], [1, 3, 16, 0, 0, 0, 1, 4, 0], [1, 3, 16, 0, 1, 0, 0, 1, 0], [0, 0, 12, 0, 0, 1, 0, 1, 0], [1, 3, 12, 0, 0, 1, 0, 2, 0], [1, 3, 12, 45, 0, 0, 1, 3, 0], [0, 0, 12, 45, 1, 0, 1, 1, 0], [1, 0, 12, 0, 1, 1, 1, 5, 1], [1, 0, 12, 0, 0, 1, 1, 5, 0], [0, 3, 12, 0, 1, 0, 1, 2, 0], [1, 0, 16, 0, 1, 0, 1, 5, 0], [0, 3, 12, 45, 1, 1, 1, 3, 0], [1, 0, 16, 0, 0, 1, 1, 1, 0], [1, 0, 12, 45, 1, 1, 0, 1, 1], [0, 0, 12, 45, 1, 0, 1, 2, 0], [1, 0, 16, 45, 1, 1, 0, 4, 0], [0, 0, 16, 0, 1, 0, 0, 4, 0], [1, 3, 16, 0, 0, 1, 0, 4, 1], [0, 0, 16, 45, 1, 0, 1, 4, 1], [0, 3, 16, 45, 1, 1, 1, 5, 0], [1, 0, 16, 0, 0, 0, 1, 4, 1], [0, 0, 12, 0, 0, 0, 0, 5, 1], [1, 3, 16, 0, 0, 0, 1, 3, 0], [1, 3, 12, 45, 0, 0, 0, 2, 0], [0, 0, 16, 45, 1, 1, 1, 4, 1], [0, 3, 16, 0, 0, 1, 1, 3, 0], [0, 0, 12, 45, 1, 1, 0, 1, 1], [0, 3, 16, 0, 0, 0, 1, 5, 0], [0, 3, 16, 0, 1, 0, 1, 1, 0], [0, 3, 12, 0, 1, 0, 0, 4, 1], [1, 0, 12, 0, 1, 1, 1, 5, 0], [0, 0, 12, 0, 0, 0, 1, 3, 0], [1, 0, 16, 0, 0, 0, 0, 5, 1], [1, 3, 16, 0, 0, 0, 0, 4, 0], [1, 0, 12, 0, 0, 1, 1, 5, 1], [0, 0, 12, 45, 1, 1, 0, 3, 0], [0, 0, 16, 45, 0, 1, 1, 1, 1], [1, 3, 16, 45, 0, 0, 0, 3, 1], [1, 3, 12, 45, 1, 0, 1, 2, 0], [1, 3, 12, 0, 0, 0, 1, 1, 0], [0, 3, 12, 45, 1, 1, 0, 5, 1], [1, 0, 12, 45, 0, 1, 1, 1, 0], [1, 0, 16, 45, 1, 1, 1, 2, 1], [1, 0, 16, 45, 1, 0, 1, 3, 1], [0, 3, 12, 45, 1, 0, 0, 5, 1], [1, 3, 12, 0, 1, 0, 0, 2, 1], [1, 3, 16, 45, 0, 1, 0, 2, 1], [1, 3, 16, 45, 1, 0, 1, 2, 1], [0, 3, 16, 45, 1, 1, 1, 5, 0], [1, 0, 12, 45, 1, 1, 1, 4, 0], [0, 3, 16, 0, 1, 1, 1, 3, 1], [0, 0, 16, 0, 1, 0, 1, 5, 1], [1, 0, 16, 45, 0, 1, 0, 1, 0], [0, 0, 12, 0, 0, 0, 1, 1, 0], [1, 0, 12, 45, 0, 1, 1, 2, 0], [1, 3, 12, 0, 1, 1, 0, 4, 0], [1, 0, 16, 0, 0, 0, 0, 4, 0], [0, 0, 12, 0, 1, 1, 1, 4, 1], [1, 0, 16, 45, 1, 1, 1, 3, 1], [0, 0, 12, 45, 1, 0, 1, 3, 1], [1, 3, 16, 0, 1, 1, 0, 4, 0], [0, 3, 12, 45, 0, 0, 0, 4, 0], [1, 3, 16, 0, 0, 0, 0, 1, 0], [1, 0, 12, 45, 1, 0, 0, 3, 0], [0, 3, 16, 45, 1, 1, 1, 4, 1], [0, 0, 12, 0, 1, 1, 0, 1, 0], [0, 3, 16, 0, 1, 1, 1, 3, 1], [1, 3, 12, 45, 0, 0, 0, 2, 0], [1, 0, 12, 45, 1, 1, 1, 1, 1], [1, 3, 16, 0, 1, 1, 0, 5, 1], [1, 0, 12, 0, 1, 0, 1, 2, 0], [0, 0, 16, 0, 0, 1, 0, 1, 1], [0, 0, 16, 0, 1, 0, 1, 4, 1], [0, 3, 12, 0, 0, 0, 0, 4, 1], [1, 0, 12, 45, 1, 0, 1, 1, 1], [0, 3, 12, 0, 1, 1, 0, 4, 0], [0, 0, 16, 45, 1, 1, 1, 1, 0], [1, 3, 16, 0, 1, 0, 1, 3, 1], [13, 21, 4, 180, 1, 1, 1, 0, 1], [12, 23, 4, 180, 0, 0, 1, 0, 0], [12, 22, 4, 180, 0, 1, 0, 0, 1], [12, 22, 4, 135, 0, 1, 0, 0, 0], [11, 23, 4, 135, 0, 0, 0, 0, 1], [11, 23, 4, 180, 0, 1, 1, 0, 1], [12, 23, 0, 180, 0, 1, 1, 0, 1], [11, 23, 4, 180, 0, 1, 0, 0, 0], [11, 22, 4, 135, 0, 0, 1, 0, 0], [12, 22, 4, 135, 1, 0, 1, 0, 1], [12, 22, 4, 180, 1, 0, 0, 0, 0], [11, 21, 4, 135, 1, 1, 0, 0, 0], [13, 21, 0, 135, 1, 0, 0, 0, 0], [12, 21, 4, 180, 1, 0, 1, 0, 1], [13, 21, 0, 135, 0, 0, 0, 0, 0], [12, 22, 0, 135, 1, 0, 0, 0, 1], [11, 21, 4, 180, 1, 1, 0, 0, 0], [12, 21, 4, 135, 0, 0, 0, 0, 0], [13, 22, 0, 135, 0, 1, 0, 0, 1], [11, 23, 4, 180, 1, 1, 1, 0, 0], [13, 22, 0, 135, 1, 0, 0, 0, 0], [11, 23, 0, 135, 0, 0, 0, 0, 0], [11, 23, 4, 135, 0, 0, 0, 0, 0], [13, 23, 0, 135, 0, 0, 0, 0, 0], [12, 23, 0, 135, 0, 0, 1, 0, 0], [12, 23, 0, 135, 1, 0, 1, 0, 0], [13, 21, 0, 180, 0, 1, 0, 0, 0], [12, 21, 4, 180, 0, 1, 0, 0, 0], [11, 23, 0, 135, 0, 0, 0, 0, 0], [12, 21, 0, 180, 1, 1, 1, 0, 0], [13, 21, 0, 135, 0, 1, 0, 0, 1], [12, 23, 4, 180, 0, 0, 1, 0, 0], [12, 22, 4, 135, 0, 0, 0, 0, 0], [13, 21, 0, 135, 1, 0, 1, 0, 0], [13, 22, 0, 180, 0, 0, 0, 0, 1], [12, 22, 4, 135, 1, 1, 0, 0, 1], [11, 23, 4, 135, 1, 1, 1, 0, 1], [13, 22, 4, 135, 0, 0, 1, 0, 1], [11, 22, 4, 180, 1, 1, 0, 0, 0], [12, 23, 4, 135, 0, 1, 0, 0, 0], [13, 22, 0, 135, 0, 0, 0, 0, 0], [12, 22, 0, 180, 1, 1, 1, 0, 0], [12, 23, 4, 135, 0, 1, 0, 0, 0], [13, 22, 0, 135, 1, 1, 0, 0, 0], [13, 22, 4, 135, 1, 0, 1, 0, 1], [11, 22, 4, 180, 1, 1, 0, 0, 0], [11, 21, 0, 135, 1, 0, 0, 0, 0], [13, 21, 0, 180, 1, 0, 0, 0, 1], [13, 23, 4, 180, 1, 0, 1, 0, 0], [11, 23, 0, 180, 0, 1, 1, 0, 1], [13, 22, 4, 135, 0, 0, 1, 0, 0], [13, 22, 4, 135, 0, 0, 1, 0, 0], [12, 23, 0, 180, 0, 1, 1, 0, 1], [12, 21, 0, 135, 0, 0, 0, 0, 0], [11, 23, 4, 135, 0, 1, 1, 0, 1], [13, 21, 4, 180, 0, 1, 0, 0, 1], [13, 22, 4, 135, 1, 1, 1, 0, 1], [13, 23, 4, 135, 0, 0, 1, 0, 1], [12, 22, 4, 135, 0, 1, 1, 0, 1], [11, 22, 0, 135, 1, 1, 1, 0, 0], [11, 21, 0, 180, 0, 1, 1, 0, 0], [13, 21, 0, 135, 1, 0, 1, 0, 1], [12, 21, 4, 180, 0, 0, 0, 0, 0], [11, 21, 0, 180, 0, 1, 1, 0, 0], [12, 23, 4, 180, 1, 1, 0, 0, 1], [13, 23, 4, 135, 1, 1, 0, 0, 0], [13, 22, 0, 135, 0, 0, 1, 0, 1], [11, 21, 0, 135, 1, 0, 1, 0, 0], [12, 23, 0, 135, 1, 1, 1, 0, 1], [12, 23, 4, 135, 0, 0, 1, 0, 1], [12, 23, 4, 135, 1, 1, 1, 0, 1], [12, 21, 0, 135, 1, 1, 0, 0, 1], [13, 21, 4, 135, 0, 0, 0, 0, 0], [11, 23, 4, 135, 0, 1, 0, 0, 1], [12, 22, 0, 180, 1, 0, 0, 0, 1], [12, 21, 4, 180, 0, 0, 1, 0, 0], [13, 23, 4, 135, 0, 1, 0, 0, 0], [12, 23, 4, 135, 1, 0, 1, 0, 0], [11, 21, 0, 180, 0, 0, 0, 0, 0], [12, 23, 4, 135, 0, 1, 1, 0, 0], [12, 23, 0, 180, 0, 0, 1, 0, 0], [11, 21, 4, 135, 0, 0, 0, 0, 1], [13, 22, 0, 180, 1, 1, 1, 0, 1], [12, 21, 4, 180, 1, 0, 0, 0, 1], [13, 23, 4, 180, 1, 0, 0, 0, 0], [11, 23, 0, 180, 0, 1, 1, 0, 0], [13, 22, 4, 180, 0, 0, 0, 0, 0], [12, 22, 4, 135, 0, 0, 1, 0, 1], [13, 23, 4, 135, 0, 1, 1, 0, 1], [13, 23, 0, 180, 0, 0, 1, 0, 0], [11, 23, 4, 135, 1, 0, 1, 0, 1], [12, 21, 4, 135, 0, 0, 1, 0, 1], [13, 23, 4, 180, 0, 1, 1, 0, 0], [13, 22, 0, 135, 0, 1, 0, 0, 0], [13, 21, 0, 180, 1, 1, 0, 0, 0], [12, 23, 0, 180, 1, 0, 1, 0, 0], [13, 23, 4, 180, 0, 1, 1, 0, 0], [12, 22, 4, 135, 0, 1, 1, 0, 1], [12, 22, 4, 180, 1, 1, 0, 0, 0], [13, 23, 0, 180, 0, 0, 1, 0, 0], [11, 22, 4, 180, 0, 1, 0, 0, 1], [13, 21, 4, 135, 0, 0, 1, 0, 1], [11, 21, 0, 135, 0, 1, 1, 0, 0], [11, 21, 4, 180, 0, 1, 1, 0, 0], [11, 23, 0, 135, 0, 0, 0, 0, 0], [12, 22, 0, 135, 0, 1, 1, 0, 0], [12, 22, 0, 135, 1, 1, 1, 0, 0], [11, 23, 0, 135, 1, 0, 0, 0, 1], [12, 22, 0, 180, 0, 0, 1, 0, 1], [11, 23, 4, 135, 0, 0, 1, 0, 1], [13, 23, 4, 180, 1, 0, 1, 0, 1], [11, 21, 0, 135, 1, 0, 0, 0, 0], [13, 21, 0, 180, 0, 1, 1, 0, 1], [12, 22, 0, 135, 1, 0, 1, 0, 1], [13, 21, 0, 135, 1, 0, 1, 0, 0], [13, 21, 0, 180, 0, 0, 0, 0, 0], [13, 23, 4, 135, 0, 0, 0, 0, 1], [11, 21, 4, 180, 0, 0, 1, 0, 1], [11, 22, 0, 180, 1, 1, 0, 0, 1], [12, 23, 4, 135, 1, 1, 1, 0, 1], [12, 23, 4, 135, 0, 0, 0, 0, 0], [13, 22, 0, 180, 1, 0, 1, 0, 0], [11, 21, 4, 135, 1, 0, 0, 0, 1], [12, 23, 4, 135, 0, 0, 0, 0, 1], [13, 21, 0, 135, 0, 1, 0, 0, 1], [11, 23, 4, 180, 0, 0, 0, 0, 1], [12, 22, 0, 135, 1, 0, 0, 0, 1], [12, 23, 0, 180, 0, 0, 0, 0, 1], [11, 23, 0, 180, 0, 1, 1, 0, 1], [11, 22, 4, 135, 1, 1, 1, 0, 1], [11, 23, 0, 180, 0, 1, 1, 0, 0], [13, 22, 0, 180, 0, 1, 0, 0, 0], [13, 22, 4, 180, 0, 0, 0, 0, 0], [11, 22, 0, 180, 0, 1, 0, 0, 1], [13, 22, 4, 180, 1, 1, 0, 0, 0], [12, 22, 0, 180, 0, 0, 1, 0, 0], [12, 22, 0, 135, 1, 0, 1, 0, 1], [12, 22, 0, 180, 1, 1, 0, 0, 1], [11, 22, 4, 135, 1, 1, 1, 0, 1], [12, 23, 0, 135, 0, 1, 1, 0, 0], [11, 21, 0, 180, 0, 0, 0, 0, 1], [13, 23, 0, 180, 1, 1, 1, 0, 1], [13, 21, 0, 135, 1, 1, 0, 0, 1], [11, 21, 4, 135, 1, 0, 0, 0, 1], [12, 21, 4, 135, 0, 1, 0, 0, 1], [11, 22, 0, 135, 1, 1, 0, 0, 0], [12, 22, 0, 180, 0, 0, 1, 0, 1], [12, 22, 0, 135, 0, 1, 0, 0, 0], [12, 22, 4, 180, 0, 1, 0, 0, 0], [12, 23, 4, 135, 0, 1, 0, 0, 1]])
testing_target_tensor = torch.tensor([[0.2766666666666667], [0.5700000000000001], [0.85], [0.3133333333333333], [0.31], [0.7033333333333334], [0.18], [0.24666666666666667], [0.21333333333333332], [0.61], [0.3366666666666667], [0.22666666666666666], [0.7133333333333334], [0.31], [0.5766666666666668], [0.53], [0.6133333333333334], [0.5233333333333334], [0.5700000000000001], [0.5900000000000001], [0.6466666666666666], [0.6799999999999999], [0.53], [0.21333333333333332], [0.24666666666666667], [0.46333333333333326], [0.32999999999999996], [0.39666666666666667], [0.20333333333333337], [0.48], [0.2866666666666667], [0.85], [0.67], [0.44666666666666666], [0.44666666666666666], [0.6466666666666667], [0.63], [0.77], [0.2800000000000001], [0.4033333333333333], [0.4533333333333333], [0.44666666666666666], [0.6633333333333333], [0.6666666666666666], [0.5433333333333333], [0.29333333333333333], [0.8166666666666668], [0.53], [0.22999999999999998], [0.6766666666666665], [0.58], [0.7566666666666667], [0.42333333333333345], [0.6933333333333334], [0.5133333333333334], [0.22666666666666666], [0.3433333333333333], [0.4833333333333333], [0.7833333333333334], [0.38666666666666666], [0.3633333333333334], [0.9066666666666666], [0.63], [0.38], [0.58], [0.2533333333333333], [0.7566666666666666], [0.21333333333333332], [0.58], [0.2966666666666667], [0.5133333333333334], [0.63], [0.38333333333333336], [0.7133333333333334], [0.6433333333333333], [0.37], [0.7633333333333334], [0.4666666666666666], [0.39666666666666667], [0.38], [0.5533333333333333], [0.6466666666666667], [0.3466666666666667], [0.47333333333333344], [0.36000000000000004], [0.43], [0.5933333333333334], [0.36], [0.5333333333333334], [0.65], [0.44666666666666666], [0.4033333333333333], [0.32666666666666666], [0.6366666666666667], [0.30333333333333334], [0.8066666666666666], [0.58], [0.23666666666666666], [0.5766666666666668], [0.6133333333333334], [0.5366666666666666], [0.73], [0.5166666666666666], [0.6033333333333333], [0.3633333333333334], [0.6033333333333333], [0.4966666666666666], [0.5133333333333334], [0.6133333333333333], [0.17666666666666667], [0.21000000000000002], [0.4033333333333333], [0.36333333333333334], [0.45666666666666667], [0.8333333333333334], [0.3466666666666667], [0.27], [0.48], [0.7633333333333334], [0.32], [0.26333333333333336], [0.9], [0.46333333333333326], [0.5700000000000001], [0.7533333333333334], [0.21333333333333332], [0.36333333333333334], [0.8400000000000001], [0.5633333333333334], [0.21333333333333337], [0.44666666666666666], [0.6733333333333333], [0.6133333333333334], [0.6599999999999999], [0.4533333333333333], [0.8133333333333332], [0.6033333333333333], [0.7566666666666667], [0.61], [0.39333333333333337], [0.21333333333333332], [0.7466666666666666], [0.38], [0.3466666666666667], [0.32], [0.6133333333333334], [0.21333333333333337], [0.14666666666666667], [0.43], [0.5433333333333333], [0.8233333333333335], [0.9666666666666666], [0.44000000000000006], [0.54], [0.5333333333333333], [0.6066666666666667], [0.55], [0.8733333333333334], [0.8066666666666666], [0.6233333333333333], [0.5833333333333333], [0.64], [0.8733333333333334], [0.6066666666666667], [0.6], [0.7733333333333333], [0.64], [0.79], [0.4566666666666667], [0.8400000000000001], [0.6], [0.5833333333333333], [0.65], [0.8166666666666668], [0.74], [0.9166666666666667], [0.9], [0.74], [0.5833333333333333], [0.54], [0.5166666666666667], [0.6], [0.8666666666666666], [1.0], [0.9], [0.8400000000000001], [0.42333333333333345], [0.6833333333333333], [0.7833333333333334], [0.6066666666666667], [0.6066666666666667], [0.6], [0.5], [0.74], [0.7733333333333333], [0.8066666666666666], [0.7233333333333334], [0.7833333333333334], [0.9833333333333332], [0.6233333333333333], [0.85], [0.9], [0.95], [0.85], [0.9233333333333335], [0.5066666666666666], [0.7233333333333334], [0.5666666666666667], [0.7833333333333334], [0.6], [0.8333333333333333], [0.95], [0.95], [0.6233333333333333], [0.9233333333333335], [0.8666666666666666], [0.7233333333333334], [0.8566666666666667], [0.4833333333333334], [0.54], [1.0], [0.74], [0.49000000000000005], [0.49000000000000005], [0.4566666666666667], [0.6], [0.5233333333333334], [0.49000000000000005], [0.6], [0.9833333333333332], [0.65], [0.54], [0.5833333333333333], [0.64], [0.65], [0.9166666666666667], [0.6], [0.6166666666666667], [0.5333333333333333], [0.8233333333333335], [0.9], [0.64], [0.85], [0.8], [0.49000000000000005], [0.42333333333333345], [1.0], [0.6066666666666667], [0.9333333333333332], [0.5066666666666666], [0.54], [0.7733333333333333], [0.6833333333333333], [0.5666666666666667], [0.7833333333333334], [0.42333333333333345], [0.95], [0.6333333333333333], [0.9666666666666666], [0.7233333333333334], [0.5233333333333333], [0.49000000000000005], [0.8733333333333334], [0.65], [0.9833333333333332], [0.8833333333333332], [0.7733333333333333], [0.7566666666666667], [0.8233333333333335], [0.8233333333333335], [0.64], [0.6833333333333333], [0.9400000000000001], [0.9], [0.5833333333333334], [0.5666666666666667], [0.65], [0.5566666666666668], [0.4833333333333334], [1.0], [0.9833333333333332], [0.9], [0.5566666666666668], [0.44000000000000006], [0.42333333333333345], [0.5166666666666667], [0.9400000000000001], [0.6333333333333333], [0.9400000000000001], [0.42333333333333345], [0.9833333333333332], [0.8566666666666667], [0.5833333333333333], [0.9], [0.9], [0.74], [0.8833333333333332], [0.5733333333333334], [0.7], [0.8233333333333335], [0.5466666666666666], [0.11333333333333333], [0.44666666666666666], [0.14666666666666667], [0.3466666666666667], [0.5133333333333334], [0.5133333333333334], [0.14666666666666667], [0.11333333333333333], [0.44666666666666666], [0.08], [0.18], [0.08], [0.44666666666666666], [0.04666666666666667], [0.38], [0.18], [0.04666666666666667], [0.44666666666666666], [0.24666666666666667], [0.08], [0.04666666666666667], [0.04666666666666667], [0.04666666666666667], [0.11333333333333333], [0.14666666666666667], [0.14666666666666667], [0.14666666666666667], [0.04666666666666667], [0.24666666666666667], [0.44666666666666666], [0.11333333333333333], [0.04666666666666667], [0.14666666666666667], [0.3466666666666667], [0.48], [0.5466666666666666], [0.4133333333333334], [0.18], [0.14666666666666667], [0.04666666666666667], [0.24666666666666667], [0.14666666666666667], [0.18], [0.44666666666666666], [0.18], [0.08], [0.38], [0.14666666666666667], [0.5133333333333334], [0.11333333333333333], [0.11333333333333333], [0.5133333333333334], [0.04666666666666667], [0.5133333333333334], [0.44666666666666666], [0.5466666666666666], [0.4133333333333334], [0.5133333333333334], [0.24666666666666667], [0.21333333333333337], [0.44666666666666666], [0.04666666666666667], [0.21333333333333337], [0.48], [0.18], [0.4133333333333334], [0.14666666666666667], [0.5466666666666666], [0.4133333333333334], [0.5466666666666666], [0.48], [0.04666666666666667], [0.44666666666666666], [0.38], [0.11333333333333333], [0.14666666666666667], [0.14666666666666667], [0.04666666666666667], [0.21333333333333337], [0.11333333333333333], [0.3466666666666667], [0.5466666666666666], [0.38], [0.08], [0.21333333333333337], [0.04666666666666667], [0.4133333333333334], [0.5133333333333334], [0.11333333333333333], [0.44666666666666666], [0.4133333333333334], [0.21333333333333337], [0.14666666666666667], [0.18], [0.14666666666666667], [0.21333333333333337], [0.5133333333333334], [0.18], [0.11333333333333333], [0.44666666666666666], [0.4133333333333334], [0.21333333333333337], [0.21333333333333337], [0.04666666666666667], [0.21333333333333337], [0.24666666666666667], [0.38], [0.4133333333333334], [0.4133333333333334], [0.44666666666666666], [0.08], [0.5133333333333334], [0.44666666666666666], [0.14666666666666667], [0.04666666666666667], [0.3466666666666667], [0.4133333333333334], [0.48], [0.5466666666666666], [0.04666666666666667], [0.14666666666666667], [0.38], [0.3466666666666667], [0.44666666666666666], [0.3466666666666667], [0.38], [0.3466666666666667], [0.5133333333333334], [0.5466666666666666], [0.21333333333333337], [0.14666666666666667], [0.04666666666666667], [0.44666666666666666], [0.18], [0.11333333333333333], [0.44666666666666666], [0.48], [0.5466666666666666], [0.21333333333333337], [0.3466666666666667], [0.5466666666666666], [0.48], [0.38], [0.44666666666666666], [0.18], [0.4133333333333334], [0.14666666666666667], [0.14666666666666667], [0.44666666666666666]])