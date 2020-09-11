import tensorflow as tf
from keras import Input, Model, regularizers
from keras.utils.generic_utils import has_arg
from keras.layers import StackedRNNCells, Dense, Concatenate

from keras_yamnet.yamnet import YAMNet


def x_calculation(tensors):
    # print("\n\n INPUT x_calculation", tensors)
    feature_img = tensors[0]
    # print("x_calculation", feature_img.shape[0])
    map_att = tensors[1]
    product = feature_img * tf.expand_dims(map_att, 2)
    return tf.reduce_sum(product, 1)


class StackedCellFeedback(StackedRNNCells):

    def __init__(self, cells, feature_shape, weight_decay=1e-5, audio_shape=(1582,), dim=0, yam_shape=None, **kwargs):
        super().__init__(cells, **kwargs)
        self.attention_maps = []
        self.attention_layers = [
            Dense(feature_shape[1], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            Dense(feature_shape[0], activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))]
        self.current_map = None
        self.feature_shape = feature_shape
        self.attention_maps = []
        self.time = 0
        self.audio_shape = audio_shape
        self.audio_tensors = []
        self.dim = dim
        self.yam_shape = yam_shape

    def attention_model(self, inputs):
        out = self.attention_layers[0](inputs)
        out = self.attention_layers[1](out)
        return out

    def call(self, inputs, states, constants=None, **kwargs):
        if self.time == 0:
            random_state = states[0]
            first_map = self.attention_model(random_state)
            self.current_map = first_map
            self.attention_maps.append(first_map)
            self.time += 1
        nested_states = []
        for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append([states[0]])
                states = states[1:]
        if self.reverse_state_order:
            nested_states = nested_states[::-1]

        # Call the cells in order and store the returned states.
        # todo qui bisogna intervenire con la mappa
        new_nested_states = []
        counter_cells = 0
        for cell, states in zip(self.cells, nested_states):
            if counter_cells == 0:
                inputs = x_calculation([inputs, self.current_map])
                if self.dim >= 0:
                    if self.yam_shape is not None:

                        yn = YAMNet(weights='keras_yamnet/yamnet_conv.h5', classes=7, classifier_activation='softmax',
                                    input_shape=(self.yam_shape, 64))
                        yamnet = Model(input=yn.input, output=yn.layers[-3].output)
                        self.audio_tensors.append(yamnet.input)
                        audio_input = yamnet.output
                    else:
                        audio_input = Input(shape=self.audio_shape)
                        self.audio_tensors.append(audio_input)
                    inputs = Concatenate(name='fusion1')([inputs, audio_input])
                    if self.dim == 0 or self.dim == 3:
                        inputs = Dense(2048, activation='relu', name='fusion2')(inputs)
            if has_arg(cell.call, 'constants'):
                inputs, states = cell.call(inputs, states,
                                           constants=constants,
                                           **kwargs)

            else:
                inputs, states = cell.call(inputs, states, **kwargs)
            new_nested_states.append(states)
            if counter_cells == len(self.cells) - 1:
                out = self.attention_model(inputs)
                self.attention_maps.append(out)
                self.current_map = out

            counter_cells += 1
        # Format the new states as a flat list
        # in reverse cell order.
        new_states = []
        if self.reverse_state_order:
            new_nested_states = new_nested_states[::-1]
        for cell_states in new_nested_states:
            new_states += cell_states
        return inputs, new_states
