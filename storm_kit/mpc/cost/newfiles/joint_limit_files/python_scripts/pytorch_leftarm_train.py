import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio


np.random.seed(0)

# Load the data
xyall = np.load('../arrays/xyall.npy')
length = xyall.shape[0]

xy_train = xyall[0:int(length*0.8), :]
xy_test = xyall[int(length*0.8):, :]

y_train = xy_train[:, 6].reshape(-1, 1).astype(int)
x_train = xy_train[:, 0:6]

y_test = xy_test[:, 6].reshape(-1, 1).astype(int)
x_test = xy_test[:, 0:6]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# should be
# [-0.06125642  0.99812206 -0.99416399 -0.10787939  0.69672186  0.13871641  1.        ]
# [-0.9572976   0.28910432  0.76661634 -0.64210543  0.12628844  0.88909463  0.        ]
print(xyall[0])
print(xyall[-1])

# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_train).float().to('cuda')
y_train = torch.from_numpy(y_train).float().to('cuda')
x_test = torch.from_numpy(x_test).float().to('cuda')
y_test = torch.from_numpy(y_test).float().to('cuda')

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

        # constant_value = 0.5
        # nn.init.constant_(self.fc1.weight, constant_value)
        # nn.init.constant_(self.fc1.bias, constant_value)
        # nn.init.constant_(self.fc2.weight, constant_value)
        # nn.init.constant_(self.fc2.bias, constant_value)
        # nn.init.constant_(self.fc3.weight, constant_value)
        # nn.init.constant_(self.fc3.bias, constant_value)
        # nn.init.constant_(self.fc4.weight, constant_value)
        # nn.init.constant_(self.fc4.bias, constant_value)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Instantiate the model, loss function, and optimizer
model = NeuralNetwork().to('cuda')
#criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), alpha=0.9) # https://stackoverflow.com/questions/72434215/rmsprop-in-tf-vs-pytorch
loss_fn = nn.BCELoss()

batch_size = 256
epochs = 60

# Train the model
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_train_batch = x_train[i:i+batch_size]
        y_pred = model(x_train_batch)
        y_train_batch = y_train[i:i+batch_size]
        loss = loss_fn(y_pred, y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}')
    print(f'latest loss {loss}')


# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(x_test)
  
accuracy = (outputs.round() == y_test).float().mean()
print(f'accuracy: {accuracy}')

# Save the model
torch.save(model.state_dict(), '../weights/3-128-ran-larm.pt')
