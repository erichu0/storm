import scipy.io as sio
import numpy as np


mat_contents = sio.loadmat('./python_scripts/randomsin_arm_left.mat')
xyall = mat_contents['qTrain_ba_sin']
print(xyall.shape)
np.random.shuffle(xyall)

np.save("./arrays/xyall", xyall)