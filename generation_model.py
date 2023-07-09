from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import activations

def create_model():
    model = Sequential()

    model.add(Input(shape=(630,)))
    model.add(Dense(1024, activation = activations.sigmoid))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation = activations.sigmoid))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation = activations.sigmoid))
    model.add(Dropout(0.25))
    model.add(Dense(24, activation = activations.softmax))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')
    # model.load_weights(".\weights\weights_ANN.h5")
    return model