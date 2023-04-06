import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import SimpleRNN,Dense

def model_fit(train_x, train_y, test_x, test_y):

    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(1, 4), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    history=model.fit(train_x, train_y, epochs=10)
    
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print('loss: {:.4f}\nacc: {:.4f}'.format(test_loss, test_acc))