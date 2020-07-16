from keras.layers import RNN, Reshape, Input
import tensorflow as tf
from keras import regularizers
import keras.backend as K
from keras.utils.generic_utils import to_list, unpack_singleton, has_arg
from keras.engine import InputSpec
from keras.engine.base_layer import _collect_input_shape
from Models.StackedCellFeedback import StackedCellFeedback
from keras.layers.recurrent import _standardize_args
from keras.utils.generic_utils import to_list, unpack_singleton, has_arg
import traceback


class RNNStackedAttention(RNN):

    def __init__(self, input_shape, cell, return_sequences=False, return_state=False, go_backwards=False,
                 stateful=False, unroll=False, audio_shape=(1582,), dim=0, **kwargs):
        self.shape_lstm = (None, input_shape[0], input_shape[2])
        super().__init__(cell, return_sequences, return_state, go_backwards, stateful, unroll, **kwargs)

        self.cell = StackedCellFeedback(cell, (input_shape[1], input_shape[2]), audio_shape=audio_shape, dim=dim)
        self.dim = dim

    def get_audio_tensors(self):
        return self.cell.audio_tensors

    def _sup_rnn_call(self, inputs, **kwargs):
        # print("\n\n\n\n_SUP_RNN_CALL")
        # print("inputs:", inputs, "\nkwargs:", kwargs)
        if isinstance(inputs, list):
            inputs = inputs[:]
        with K.name_scope(self.name):
            # Handle laying building (weight creating, input spec locking).
            # print("built", self.built, self.shape_lstm)
            if not self.built:
                input_shapes = [self.shape_lstm]

                # Collect input shapes to build layer.
                # input_shapes.append(self.shape_lstm)

                for x_elem in to_list(inputs):
                    # print("hasattr(x_elem, '_keras_shape'):", hasattr(x_elem, '_keras_shape'))
                    # print("\nhasattr(K, 'int_shape'):", hasattr(K, 'int_shape'))
                    if hasattr(x_elem, '_keras_shape'):
                        input_shapes.append(x_elem._keras_shape)
                    elif hasattr(K, 'int_shape'):
                        input_shapes.append(K.int_shape(x_elem))
                    else:
                        raise ValueError(
                            'You tried to call layer "' + self.name + '".\n' +
                            'This layer has no information about its expected input shape, and thus cannot be built.' +
                            'You can build it manually via: `layer.build(batch_input_shape)`')

                # print("CALL BUILD", "\n", input_shapes, "\n", input_shapes[0], "\ncall:")
                if self.dim > 0:
                    input_shapes[0] = (None, 3, 3630)
                self.build(unpack_singleton(input_shapes))
                self.built = True

                # Load weights that were specified at layer instantiation.
                # print("self._initial_weights:", self._initial_weights)
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)

            # Raise exceptions in case the input is not compatible
            # with the input_spec set at build time.
            # self.assert_input_compatibility(inputs)

            # Handle mask propagation.
            user_kwargs = kwargs.copy()

            # Actually call the layer,
            # collecting output(s), mask(s), and shape(s).
            # print("call CALL:\n  inputs:", inputs, "  \nkwargs", kwargs)
            output = self.call(inputs, **kwargs)
            # print("call OUTPUT:", output)
            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = to_list(output)
            inputs_ls = to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if id(x) in [id(i) for i in inputs_ls]:
                    x = K.identity(x)
                output_ls_copy.append(x)
            output = unpack_singleton(output_ls_copy)

            input_shape = _collect_input_shape(inputs)
            input_shape[0] = self.shape_lstm
            # Inferring the output shape is only relevant for Theano.
            if all([s is not None
                    for s in to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
                # print("output_shape 1", output_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None
            #
            # if (not isinstance(output_mask, (list, tuple)) and
            #         len(output_ls) > 1):
            #     # Augment the mask to match the length of the output.
            #     output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            previous_mask = None
            output_mask = None

            # def _add_inbound_node(self, input_tensors, output_tensors,
            #                       input_masks, output_masks,
            #                       input_shapes, output_shapes, arguments=None):
            # print("_add_inbound_node CALL:\n  inputs:", inputs, "  \noutput_tensors:", output)
            # print("  \ninput_masks:", previous_mask, "  \noutput_masks:", output_mask)
            # print("  \ninput_shapes:", input_shape, "  \noutput_shape:", output_shape)
            # print("  \nuser_kwargs:", user_kwargs)
            self._add_inbound_node(input_tensors=inputs,
                                   output_tensors=output,
                                   input_masks=previous_mask,
                                   output_masks=output_mask,
                                   input_shapes=input_shape,
                                   output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            # self has no activity_regularizer
            if (hasattr(self, 'activity_regularizer') and
                    self.activity_regularizer is not None):
                with K.name_scope('activity_regularizer'):
                    regularization_losses = [
                        self.activity_regularizer(x)
                        for x in to_list(output)]
                self.add_loss(regularization_losses,
                              inputs=to_list(inputs))
        return output

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        # print("\n\n\n\n__CALL__")
        # traceback.print_stack()
        # print("\n\n\n\n")
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)
        # print("inputs:", inputs, "\ninitial_state:", initial_state, "\nconstants:", constants)
        # print("\nself._num_constants", self._num_constants)

        if initial_state is None and constants is None:
            return self._sup_rnn_call(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []

        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            if 'initial_state' in kwargs:
                kwargs.pop('initial_state')
            if 'constants' in kwargs:
                kwargs.pop('constants')
            # print("sup_rnn CALL:\n  full_input:", full_input, "  \nkwargs", kwargs)
            # print("Temporary new input_spec", full_input)
            output = self._sup_rnn_call(full_input, **kwargs)
            # print("sup_rnn OUTPUT:", output)
            self.input_spec = original_input_spec
            return output
        else:
            return self._sup_rnn_call(inputs, **kwargs)

    def get_initial_state(self, inputs):
        # print("\n\n\n\nCALL get_initial_state")
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.ones_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        # print("\n\n\n\nCALL CALL")
        # print("inputs:", inputs, "\nmask:", mask, "\ntraining:", training, "\ninitial_state:", initial_state)
        # print("\nconstants:", constants)
        if not isinstance(initial_state, (list, tuple, type(None))):
            initial_state = [initial_state]
        if not isinstance(constants, (list, tuple, type(None))):
            constants = [constants]
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            if len(inputs) == 1:
                inputs = inputs[0]
            else:
                # get initial_state from full input spec
                # as they could be copied to multiple GPU.
                # print("self._num_constants:", self._num_constants)
                if self._num_constants is None:
                    if initial_state is not None:
                        raise ValueError('Layer was passed initial state ' +
                                         'via both kwarg and inputs list)')
                    initial_state = inputs[1:]
                else:
                    # print("inputs[1:-self._num_constants]:", inputs[1:-self._num_constants])
                    if initial_state is not None and inputs[1:-self._num_constants]:
                        raise ValueError('Layer was passed initial state ' +
                                         'via both kwarg and inputs list')
                    initial_state = inputs[1:-self._num_constants]
                    if constants is None:
                        constants = inputs[-self._num_constants:]
                    elif len(inputs) > 1 + len(initial_state):
                        raise ValueError('Layer was passed constants ' +
                                         'via both kwarg and inputs list)')
                if len(initial_state) == 0:
                    initial_state = None
                inputs = inputs[0]
        # print("self.stateful:", self.stateful)
        # print("CHECK initial_state:", initial_state)
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            reshape = Reshape((inputs._keras_shape[1], inputs._keras_shape[2] * inputs._keras_shape[3]))(inputs)
            initial_state = self.get_initial_state(reshape)

        if isinstance(mask, list):
            mask = mask[0]
        # print("len(initial_state) != len(self.states)", len(initial_state), len(self.states))
        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')

        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        # print("has_arg(self.cell.call, 'training')", has_arg(self.cell.call, 'training'))
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                # print("------------- INPUT step CALL -------------")
                # print("  inputs:", inputs, "\n  states:", states)
                temp = self.cell.call(inputs, states, **kwargs)
                # print("------------- END step CALL -------------")
                return temp
        # print("k.rnn CALL:\n  inputs:", inputs, "  \ninitial_state:", initial_state, "  \nconstants:", constants)
        # print("  \ngo_backwards:", self.go_backwards, "  \nmask:", mask, "  \nunroll:", self.unroll)
        # print("  \ntimesteps:", timesteps)
        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        # print("K.rnn OUTPUT:\n  last_output:", last_output, "\n  outputs:", outputs, "\n  states:", states)
        # print("self.stateful", self.stateful)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)
        # print("self.return_sequences", self.return_sequences)
        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        # print("getattr(lastoutput, uses_learning_phase, False)", getattr(last_output, '_uses_learning_phase', False))
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True
        # print("self.return_state:", self.return_state)
        if self.return_state:
            states = to_list(states, allow_tuple=True)
            return [output] + states
        else:
            return output
