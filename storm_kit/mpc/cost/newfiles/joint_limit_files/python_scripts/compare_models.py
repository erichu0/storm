import scipy.io as sio
import torch
import keras
import numpy as np
import torch.nn as nn
# import keras_leftarm_train
# import pytorch_leftarm_train

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Set up testing split
xyall = np.load('../arrays/xyall.npy')
length = xyall.shape[0]

xy_test = xyall[int(length*0.8):, :]

x_test = xy_test[:, 0:6]
y_test = xy_test[:, 6].reshape(-1, 1).astype(int)


# Import models
keras_model = keras.models.load_model('../weights/3-128-ran-larm.h5')
pytorch_model = torch.load('../weights/3-128-ran-larm.pt').to('cpu')


# Predict
keras_raw = keras_model.predict(x_test)
pytorch_raw = pytorch_model(torch.from_numpy(x_test).float()).detach().numpy()

keras_pred = np.round(keras_raw)
pytorch_pred = np.round(pytorch_raw)


# Compare predictions
print(f'0s in Pytorch: {(pytorch_pred == 0).sum()}')
print(f'0s in Keras: {(keras_pred == 0).sum()}')

print(f'# of same outputs: {np.count_nonzero(pytorch_pred == keras_pred)}')
print(f'# total outputs: {xy_test.shape[0]}')
print(f'% same output: {(np.count_nonzero(pytorch_pred == keras_pred))/xy_test.shape[0] * 100}')

np.savetxt("../results/pytorch_out.txt", pytorch_pred)
np.savetxt("../results/keras_out.txt", keras_pred)
np.savetxt("../results/ground_truth.txt", y_test)

print(keras_pred[0])

print("keras confusion matrix: true positive, false positive, false negative, true negative")
keras_cm = np.array([0, 0, 0, 0])
for i in range(len(keras_pred)):
    if keras_pred[i] == 1 and y_test[i] == 1:
        keras_cm[0] += 1
    elif keras_pred[i] == 1 and y_test[i] == 0:
        keras_cm[1] += 1
    elif keras_pred[i] == 0 and y_test[i] == 1:
        keras_cm[2] += 1
    elif keras_pred[i] == 0 and y_test[i] == 0:
        keras_cm[3] += 1

print(keras_cm/len(pytorch_pred)*100)


print("pytorch confusion matrix: true positive, false positive, false negative, true negative")
pytorch_cm = np.array([0, 0, 0, 0])
for i in range(len(pytorch_pred)):
    if pytorch_pred[i] == 1 and y_test[i] == 1:
        pytorch_cm[0] += 1
    elif pytorch_pred[i] == 1 and y_test[i] == 0:
        pytorch_cm[1] += 1
    elif pytorch_pred[i] == 0 and y_test[i] == 1:
        pytorch_cm[2] += 1
    elif pytorch_pred[i] == 0 and y_test[i] == 0:
        pytorch_cm[3] += 1

print(pytorch_cm/len(pytorch_pred)*100)