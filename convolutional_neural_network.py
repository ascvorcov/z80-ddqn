from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, Add
from keras.optimizers import RMSprop
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

def huber_loss(q, y_pred):
    error = K.abs(q - y_pred)
    #clip_error = keras.backend.clip(error, 0.0, 1.0)
    linear_error = error - .5
    use_linear_flag = K.cast((error > 1), 'float32')
    return K.mean((use_linear_flag * linear_error + .5*(1-use_linear_flag) * K.square(error)))

class ConvolutionalNeuralNetwork:

    def create_original(input_shape, action_space):
        model = Sequential()
        model.add(Conv2D(32,
                         8,
                         strides=(4, 4),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        model.add(Conv2D(64,
                         4,
                         strides=(2, 2),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        model.add(Conv2D(64,
                         3,
                         strides=(1, 1),
                         padding="valid",
                         activation="relu",
                         input_shape=input_shape,
                         data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(action_space))
        model.compile(loss=huber_loss,
                      optimizer=RMSprop(lr=0.00025,
                                        rho=0.95,
                                        epsilon=0.01),
                      metrics=["accuracy"])
        return model

    def create_dueling(input_shape, action_space):
        inputs = Input(shape=input_shape)
        shared = Conv2D(32, (8, 8), strides=(4, 4), padding="valid", activation="relu", data_format="channels_first")(inputs)
        shared = Conv2D(64, (4, 4), strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(shared)
        shared = Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(shared)
        flatten = Flatten()(shared)

        advantage_fc = Dense(512, activation="relu")(flatten)
        advantage = Dense(action_space)(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(action_space,))(advantage)

        value_fc = Dense(512, activation="relu")(flatten)
        value =  Dense(1)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(action_space,))(value)

        q_value = Add()([value, advantage])
        model = Model(inputs=inputs, outputs=q_value)

        model.compile(loss=huber_loss,
                           optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                           metrics=["accuracy"])
        return model
        #plot_model(self.model, show_shapes=True, expand_nested=True, to_file='model.png')