from keras.datasets import mnist
import keras
from keras import backend as K
from architecture import Model


if __name__ == "__main__":
    batch_size = 128
    classes = 10
    epochs = 12

    height, width, channels = 28, 28, 1

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, height, width)
        x_test = x_test.reshape(x_test.shape[0], channels, height, width)
    else:
        x_train = x_train.reshape(x_train.shape[0], height, width, channels)
        x_test = x_test.reshape(x_test.shape[0], height, width, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)

    model = Model.build_network(classes, height, width, channels)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("mnist_model.h5")

