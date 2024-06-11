from tensorflow.keras import utils
import keras
import Conv_model_study.resnet.resnet18 as resnet

#mnist dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

#img preprocessing
x_train = x_train.reshape(-1,28,28,1)/255.
x_test = x_test.reshape(-1,28,28,1)/255.
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)


model = resnet.Resnet18((28,28,1), 10)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs = 10,
    validation_data = (x_test, y_test)
)

