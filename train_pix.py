# https://github.com/Kerite/Remote-Sensing-Classification/blob/main/train_pix.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, MaxPooling1D, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

TRAINING_SAMPLE = 500
data_path = r"data_set"
batch_size = 250
epochs = 20

# 读取数据集
paviaU = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
paviaUgt = sio.loadmat(os.path.join(data_path,
                                    'PaviaU_gt.mat'))['paviaU_gt'] - 1

paviaU = paviaU.reshape([paviaU.shape[0] * paviaU.shape[1], paviaU.shape[2]])
paviaUgt = paviaUgt.reshape((paviaUgt.shape[0] * paviaUgt.shape[1], 1))

count = 0
for x in range(paviaU.shape[0]):
    if paviaUgt[x][0] >= 0 and paviaUgt[x][0] < 255:
        count = count + 1

X = np.ndarray((count, paviaU.shape[1]))
Y = np.ndarray((count, 1), dtype=np.int32)
count = 0
for x in range(paviaU.shape[0]):
    if paviaUgt[x][0] >= 0 and paviaUgt[x][0] < 255:
        X[count, :] = paviaU[x, :]
        Y[count, :] = int(np.round(paviaUgt[x, :]))

        count = count + 1

print('count=', count)

CLASSES = (np.unique(Y)).shape[0]

countClass = np.ndarray((CLASSES))
for x in range(CLASSES):
    countClass[x] = 0
for x in range(Y.shape[0]):
    countClass[Y[x, 0]] = countClass[Y[x, 0]] + 1

print('No of classes', countClass)

X = X / np.max(X, axis=0)

Xtrain = np.ndarray((TRAINING_SAMPLE * CLASSES, X.shape[1]))
Ytrain = np.ndarray((TRAINING_SAMPLE * CLASSES, 1), dtype=np.int32)
count = 0
for x in range(CLASSES):
    countClass[x] = 0
while count < TRAINING_SAMPLE * CLASSES:
    r = np.random.randint(0, X.shape[0])
    if countClass[Y[r]] != TRAINING_SAMPLE:
        Xtrain[count] = X[r]
        Ytrain[count] = Y[r]
        countClass[Y[r]] = countClass[Y[r]] + 1
        count = count + 1

Y = to_categorical(Y, np.max(Y) + 1)
Ytrain = to_categorical(Ytrain, np.max(Ytrain) + 1)

XX = X
XXtrain = Xtrain
X = np.ndarray((X.shape[0], X.shape[1], 1))
Xtrain = np.ndarray((Xtrain.shape[0], Xtrain.shape[1], 1))
for i in range(len(X)):
    for j in range(len(X[0])):
        X[i, j, 0] = XX[i, j]

for i in range(len(Xtrain)):
    for j in range(len(Xtrain[0])):
        Xtrain[i, j, 0] = XXtrain[i, j]

model = Sequential()
model.add(
    Conv1D(filters=20, kernel_size=11, input_shape=(103, 1),
           activation='tanh'))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Flatten())
model.add(Dense(100, activation='tanh'))
model.add(Dense(9, activation='softmax'))

adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
#训练
history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
