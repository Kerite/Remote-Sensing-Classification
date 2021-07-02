# https://github.com/Kerite/Remote-Sensing-Classification/blob/main/train.py
import io
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback, TensorBoard
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.metrics import accuracy_score

import splitfolders

# 分割数据集（训练，验证，测试 8：1：1）
splitfolders.ratio(r'PatternNet',
                   output='dataset',
                   seed=1337,
                   ratio=(0.8, 0.1, 0.1))

# 参数
#训练的轮数
epochs = 20
#每批的样本数
batch_size = 8
#图片大小
image_size = 256
#模型保存的文件名
model_path = "model.h5"
#日志保存的文件夹（训练开始的时间）
logdir = os.path.join("pattern_log",
                      time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
if not os.path.exists(logdir):
    os.makedirs(logdir)
else:
    exit()
#用于保存混淆矩阵的Writer
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

datagen = ImageDataGenerator(rescale=1.0 / 255)
train = datagen.flow_from_directory('dataset/train',
                                    batch_size=batch_size,
                                    seed=114514)
val = datagen.flow_from_directory('dataset/val', seed=114514)
test = datagen.flow_from_directory('dataset/test', seed=114514)
print(train.image_shape)

# Get the images from a batch
imgs = train.next()
labels = list(train.class_indices.keys())

#预读取验证数据
x_val, y_val = [], []
for i in range(5):
    images, classes = next(val)
    x_val.append(images.reshape(len(images), image_size * image_size * 3))
    y_val.append(np.argmax(classes, axis=1))
x_val = np.vstack(x_val)
y_val = np.hstack(y_val)

#预读取测试数据
x_test, y_test = [], []
for i in range(5):
    images, classes = next(test)
    x_test.append(images.reshape(len(images), image_size * image_size * 3))
    y_test.append(np.argmax(classes, axis=1))
x_test = np.vstack(x_test)
y_test = np.hstack(y_test)

print("Test Shape:{},{}".format(x_test.shape, y_test.shape))
print("Val Shape:{},{}".format(x_val.shape, y_val.shape))


#记录混淆矩阵的回调
def log_confussion_matrix(epoch, log):
    print(log)
    mn_pred = model.predict(
        x_test.reshape((x_test.shape[0], image_size, image_size, 3)))
    mn_pred = np.argmax(mn_pred, axis=1)
    print("Predict accuracy:", accuracy_score(y_test, mn_pred))
    cm = confusion_matrix(y_test, mn_pred)
    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(cm,
                annot=True,
                cbar=False,
                fmt='d',
                xticklabels=labels,
                yticklabels=labels)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    with file_writer_cm.as_default():
        tf.summary.image('Confusion Matrix', image, step=epoch)


# 构建模型
model = InceptionResNetV2(include_top=True,
                          weights=None,
                          classes=len(labels),
                          input_shape=(image_size, image_size, 3))
model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss='categorical_crossentropy')

# 定义回调
# Tensorboard
tensorboard_callback = TensorBoard(log_dir=logdir,
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=False)
# 混淆矩阵
confusion_callback = LambdaCallback(on_epoch_end=log_confussion_matrix)

# 开始训练
model.fit(x=train,
          validation_data=val,
          epochs=epochs,
          callbacks=[tensorboard_callback, confusion_callback])
loss, acc = model.evaluate(test)
print("loss is {}, accuracy is {}".format(loss, acc))
model.save(model_path)
