from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant
import tensorflow as tf
import scipy.io as sio
import numpy as np

np.random.seed(0)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load data
xyall = np.load('../arrays/xyall.npy')
length = xyall.shape[0]

xy_train = xyall[0:int(length*0.8),:]
xy_test = xyall[int(length*0.8):,:]

y_train = xy_train[:,6].reshape(-1,1).astype(float)
x_train = xy_train[:,0:6]

y_test = xy_test[:,6].reshape(-1,1).astype(float)
x_test = xy_test[:,0:6]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# should be
# [-0.06125642  0.99812206 -0.99416399 -0.10787939  0.69672186  0.13871641  1.        ]
# [-0.9572976   0.28910432  0.76661634 -0.64210543  0.12628844  0.88909463  0.        ]
print(xyall[0])
print(xyall[-1])


# Train model
constant_value = 0.5
model = Sequential()
model.add(Dense(128, input_dim=6, activation='tanh',
                # kernel_initializer=Constant(constant_value),
                # bias_initializer=Constant(constant_value)
                ))
model.add(Dense(128, activation='tanh',
                # kernel_initializer=Constant(constant_value),
                # bias_initializer=Constant(constant_value)
                ))
model.add(Dense(128, activation='tanh',
                # kernel_initializer=Constant(constant_value),
                # bias_initializer=Constant(constant_value)
                ))
model.add(Dense(1, activation='sigmoid',
                # kernel_initializer=Constant(constant_value),
                # bias_initializer=Constant(constant_value)
                ))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.RMSprop(rho=0.9), # https://stackoverflow.com/questions/72434215/rmsprop-in-tf-vs-pytorch
              metrics=['accuracy'])
#print(model.metrics_names)

model.fit(x_train, y_train, epochs=60, batch_size=256, shuffle=False)
score = model.evaluate(x_test, y_test, batch_size=256)
print(score)

model.save('../weights/3-128-ran-larm.h5')