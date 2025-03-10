import keras as k
from keras import layers as l
import tensorflow as tf

# Helper function to inspect the partial model
def inspect_partial(inputs, outputs):
    model = k.Model(inputs=inputs, outputs=outputs)
    model.summary()


# Inputs

num_piece_types = 6
num_colors = 2
num_files = 8
num_ranks = 8

# +1 for the possible ep capture square.
board_input_len = num_colors * (num_piece_types + 1)
board_input = l.Input(
    shape=(num_files, num_ranks, board_input_len), 
    name="board_input"
)

# Castling rights (boolean), 
# turn (boolean), 
# plys after last capture or pawn move (should be normalized / devided by 50)
state_input_len = 4 + 1 + 1
state_input = l.Input(
    shape=(state_input_len,),
    name="state_input"
)


# Convolutional layers (for the board input)

def conv_block(x, filters, kernel_size, conv_type = l.Conv2D, padding = "same"):
    conv = conv_type(filters, kernel_size, padding=padding)(x)
    relu = l.ReLU()(conv)
    norm = l.BatchNormalization()(relu)
    return norm

def res_block(x, fn):
    return l.Add()([fn(x), x])

def conv_block_branch(x, filters, kernel_sizes, conv_type = l.Conv2D, padding = "same"):
    branches = [conv_block(x, filters, ks, conv_type, padding) for ks in kernel_sizes]
    return l.Concatenate()(branches)

def conv_block_seq(x, filters, kernel_sizes):
    for kernel_size in kernel_sizes:
        x = conv_block(x, filters, kernel_size)
    return x

# Block 1
# Board inputs are bitboards, so we don't need to normalize anything.
b1_conv_sizes = [(3, 3), (5, 5), (9, 9), (15, 15), (1,8), (8,1)]
b1_conv_filters = 16
b1_conv = lambda x: conv_block_branch(x, b1_conv_filters, b1_conv_sizes, l.Conv2D) # todo: l.SeparableConv2D ?
# todo: add a residual connection here
b1_out = b1_conv(board_input)

# Block 2
b2_conv_filters = len(b1_conv_sizes) * b1_conv_filters
b2_conv = lambda x: conv_block(x, b2_conv_filters, (1, 1))
b2_res = res_block(b1_out, b2_conv)
b2_pool = l.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(b2_res)

# Block 3
b3_conv_sizes = [(3, 3), (5, 5)]
b3_conv_filters = 32

b3_adapter_filters = len(b3_conv_sizes) * b3_conv_filters
b3_adapter = conv_block(b2_res, b3_adapter_filters, (1, 1))

b3_conv = lambda x: conv_block_branch(x, b3_conv_filters, b3_conv_sizes)
b3_res = res_block(b3_adapter, b3_conv)
b3_pool = l.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(b3_res)
b3_out = b3_pool

# Block 4
b4_conv_sizes = [(3, 3), (5, 5)]
b4_conv_filters = 32

b4_adapter_filters = len(b4_conv_sizes) * b4_conv_filters
b4_adapter = conv_block(b3_res, b4_adapter_filters, (1, 1))

b4_conv = lambda x: conv_block_branch(x, b4_conv_filters, b4_conv_sizes)
b4_res = res_block(b4_adapter, b4_conv)
b4_pool = l.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(b4_res)
b4_out = b4_pool

# Block 5
b5_conv_sizes = [(3, 3)]
b5_conv_filters = 64

b5_adapter_filters = len(b5_conv_sizes) * b5_conv_filters
b5_adapter = conv_block(b4_res, b5_adapter_filters, (1, 1))

b5_conv = lambda x: conv_block_branch(x, b5_conv_filters, b5_conv_sizes)
b5_res = res_block(b5_adapter, b5_conv)
b5_pool = l.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(b5_res)
b5_out = b5_pool

# Flatten
flattened = l.Flatten()(b5_out)


# Concatenate board cnn output and state inputs

combined = l.Concatenate()([flattened, state_input])


# Dense layers

n_units = 64 + state_input_len
for _ in range(4):
    combined = l.Dense(
        units=n_units,
        activation="relu"
    )(combined)
    n_units = n_units * 2
    

# Output layers

value_output = l.Dense(
    units=1,
    activation="tanh",
    name="value_output"
)(combined)

num_sq = num_files * num_ranks
from_to = num_sq * num_sq
num_promos = 1 + 1 # Knight and queen, ignoring bishop and rook
num_promo_moves = num_files * num_promos
num_moves = from_to + num_promo_moves

policy_output = l.Dense(
    units=num_moves,
    activation="softmax",
    name="policy_output"
)(combined)


# Model
model = k.Model(
    inputs=[board_input, state_input],
    outputs=[value_output, policy_output]
)

model.summary()

## Input specs

board_input_spec = tf.TensorSpec(shape=[None, num_files, num_ranks, board_input_len], dtype=tf.float32)
state_input_spec = tf.TensorSpec(shape=[None, state_input_len], dtype=tf.float32)

## Training

optimizer = k.optimizers.Adam()
policy_loss_fn = k.losses.CategoricalCrossentropy()
value_loss_fn = k.losses.MeanSquaredError()