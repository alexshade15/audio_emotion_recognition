from keras import Input, regularizers
from keras.layers import StackedRNNCells, Reshape, Dense, Softmax, Concatenate
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import has_arg
import traceback



def x_calculation(tensors):
    #print("\n\n INPUT x_calculation", tensors)
    feature_img = tensors[0]
    #print("x_calculation", feature_img.shape[0])
    map_att = tensors[1]
    product = feature_img * tf.expand_dims(map_att, 2)
    sum = tf.reduce_sum(product,1)
    return sum


class StackedCellFeedback(StackedRNNCells):

    def __init__(self,cells,feature_shape,weight_decay = 1e-5, feature_audio=1582):
        super(StackedCellFeedback,self).__init__(cells)
        self.attention_maps = []
        self.attention_layers = [Dense(feature_shape[1],activation='relu',kernel_regularizer=regularizers.l2(weight_decay)), Dense(feature_shape[0],activation='softmax',kernel_regularizer=regularizers.l2(weight_decay))]
        self.current_map = None
        self.feature_shape = feature_shape
        self.attention_maps = []
        self.time = 0
        self.feature_audio = feature_audio


    def attention_model(self, inputs):
        out = self.attention_layers[0](inputs)
        out = self.attention_layers[1](out)
        return out


    def call(self, inputs, states, constants=None, **kwargs):
        #print("\n\n\n------------------------ CALL from StackedCellFeedback ------------------------")
        #print("\n\nstates " + str( states) + "\nconstants " + str(constants) + "\nkwargs " +str(kwargs))
        #print("ciao")
        #reshaped_feature_for_calculation = Reshape((self.feature_shape[0], self.feature_shape[1]))(inputs)
        #print("\n\n SELF.TIME: " + str(self.time))
        if self.time == 0:
            random_state = states[0]
            first_map = self.attention_model(random_state)
            #first_map = Softmax(first_map)
            self.current_map = first_map
            self.attention_maps.append(first_map)
            self.time += 1
        # Recover per-cell states.
        #
        nested_states = []
        #print("\n\nself.reverse_state_order " + str(self.reverse_state_order))
        #print("\nself.cells " + str(self.cells))
        #print("\nlen self.cells " + str(len(self.cells)))
        for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
            #print("\n\nhasattr(cell.state_size, '__len__') " + str(hasattr(cell.state_size, '__len__')))
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append([states[0]])
                states = states[1:]
        if self.reverse_state_order:
            nested_states = nested_states[::-1]

        # Call the cells in order and store the returned states.
        #todo qui bisogna intervenire con la mappa
        new_nested_states = []
        counter_cells = 0
        for cell, states in zip(self.cells, nested_states):
            #print("\n\ncounter_cells " + str(counter_cells))
            if counter_cells == 0:
                inputs = x_calculation([inputs,self.current_map])
                #print("\nFUSION ATTENTION", inputs)
                audio_input = Input(shape=(self.feature_audio,))
                inputs = Concatenate(name='fusion1')([inputs, audio_input])
                inputs = Dense(2048, activation='relu', name='fusion2')(inputs)
                #print("§§§§§§§§§§§§§§§§§§§§§§§§------------------- FUSION AUDIO: OK -------------------§§§§§§§§§§§§§§§§§§§§§§§§")

            #print("\n\nhas_arg(cell.call, 'constants') " + str(has_arg(cell.call, 'constants')))
            if has_arg(cell.call, 'constants'):
                inputs, states = cell.call(inputs, states,
                                           constants=constants,
                                           **kwargs)

            else:
                #print("_________________________LA_CALL", type(cell), cell)
                inputs, states = cell.call(inputs, states, **kwargs)
                #print("_________________________LA_CALL END")
            new_nested_states.append(states)
            #print("\n\ncounter_cells == len(self.cells) - 1 /->/    " + str(counter_cells) + " == " + str(len(self.cells) - 1))
            #print("\nlen(cell.state_size) " + str(cell.state_size))
            if counter_cells == len(self.cells)-1:
                out = self.attention_model(inputs)
                self.attention_maps.append(out)
                self.current_map = out

            counter_cells += 1
        # Format the new states as a flat list
        # in reverse cell order.
        new_states = []
        if self.reverse_state_order:
            new_nested_states = new_nested_states[::-1]
        #print("\nnew_nested_states " + str(new_nested_states) + " " + str(len(new_nested_states)))
        for cell_states in new_nested_states:
            new_states += cell_states
        #print("\n\n\n---------------------- END CALL from StackedCellFeedback ----------------------")
        return inputs, new_states
