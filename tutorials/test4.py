import tensorflow as tf


# --- First define placeholders with fixed sizes
size_data = [32, 128, 128]
size_minibatch = 4  # number of images as one input minibatch
# only the gray-scale intensity values as one-channel feature, hence [1]:
ph_image = tf.placeholder(tf.float32, [size_minibatch]+size_data+[1])
ph_label = tf.placeholder(tf.float32, [size_minibatch]+size_data+[1])


import utilities as util

# --- usually, there is a data pre-processing (augmentation) layer
size_input = [16, 64, 64]  # use smaller images so this can be reasonably tested, e.g. on a CPU
input_image = util.resize_volume(ph_image, size_input)
input_label = util.resize_volume(ph_label, size_input)

# --- add the first block of layers (N.B. "layer" is not used consistently)
k_conv = [3, 3, 3]  # convolution kernel size
nc1 = 8  # number of feature maps after the convolution layer, i.e. channels
# -Step 1- declare variables for storing convolution kernel "weights" to optimise
W1 = tf.get_variable("W1", shape=k_conv+[1, nc1], initializer=tf.contrib.layers.xavier_initializer())
# -Step 2- layers include:
# (a) 3d convolution with padded feature maps (for convenience in working out the sizes at different resolution levels)
# (b) batch normalisation, and
# (c) nonlinear activation (relu in this case)
strides_one = [1, 1, 1, 1, 1]  # stride of the sliding window used in convolution
layer1c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(input_image, W1, strides_one, padding="SAME")))
# -Step 3- polling layer
k_pool = [1, 2, 2, 2, 1]  # kernel for pooling
layer1 = tf.nn.max_pool3d(layer1c, k_pool, strides_one, "SAME")

# --- add down-sampling convolution layers
nc2 = nc1*2  # double the number of feature maps
W2 = tf.get_variable("W2", shape=k_conv+[nc1, nc2], initializer=tf.contrib.layers.xavier_initializer())
layer2c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer1, W2, strides_one, "SAME")))
# down-sample via pooling layer
strides_two = [1, 2, 2, 2, 1]  # stride used for down-sampling and up-sampling
layer2 = tf.nn.max_pool3d(layer2c, k_pool, strides_two, "SAME")

# --- add two more of these down-sampling blocks
nc3 = nc2*2
W3 = tf.get_variable("W3", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
layer3c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer2, W3, strides_one, "SAME")))
layer3 = tf.nn.max_pool3d(layer3c, k_pool, strides_two, "SAME")

nc4 = nc3*2
W4 = tf.get_variable("W4", shape=k_conv+[nc3, nc4], initializer=tf.contrib.layers.xavier_initializer())
layer4c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer3, W4, strides_one, "SAME")))
layer4 = tf.nn.max_pool3d(layer4c, k_pool, strides_two, "SAME")

# --- add a convolution layer without sampling/pooling at the end of down-sampling blocks (i.e. the "encoder")
nc5 = nc4*2
W5 = tf.get_variable("W5", shape=k_conv+[nc4, nc5], initializer=tf.contrib.layers.xavier_initializer())
layer5 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer4, W5, strides_one, "SAME")))


# --- add an up-sampling block (starting the "decoder")
# (a) the number of feature maps are halved;
# (b) use transpose convolution for up-sampling;
# (c) make sure the output size is the same as that of layer4c so they can be added (or concatenated)
W4d = tf.get_variable("W4d", shape=k_conv+[nc4, nc5], initializer=tf.contrib.layers.xavier_initializer())
layer4d = tf.nn.conv3d_transpose(layer5, W4d, output_shape=layer4c.get_shape(), strides=strides_two, padding="SAME")
# (d) add a skip layer (shortcut connection) to the encoder using summation
# (e) then add another convolution, batch normalisation and nonlinear activation
W4dc = tf.get_variable("W4dc", shape=k_conv+[nc4, nc4], initializer=tf.contrib.layers.xavier_initializer())
layer4dc = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer4d+layer4c, W4dc, strides_one, "SAME")))

# --- add another three of these up-sampling blocks, till the original input size
W3d = tf.get_variable("W3d", shape=k_conv+[nc3, nc4], initializer=tf.contrib.layers.xavier_initializer())
layer3d = tf.nn.conv3d_transpose(layer4dc, W3d, layer3c.get_shape(), strides_two, "SAME")
W3dc = tf.get_variable("W3dc", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
layer3dc = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer3d+layer3c, W3dc, strides_one, "SAME")))

W2d = tf.get_variable("W2d", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
layer2d = tf.nn.conv3d_transpose(layer3dc, W2d, layer2c.get_shape(), strides_two, "SAME")
W2dc = tf.get_variable("W2dc", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
layer2dc = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer2d+layer2c, W2dc, strides_one, "SAME")))


# --- add the "read-out" layer with a sigmoid activation
# use single channel for soft Dice loss
W1d = tf.get_variable("W_out", shape=k_conv+[nc2, 1], initializer=tf.contrib.layers.xavier_initializer())
layer1d = tf.sigmoid(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer2dc, W1d, strides_one, "SAME")))

# check if output size is expected
print(layer1d)


# loss based on Dice, between predicted segmentation layer_out and ground-truth label
dice_numerator = tf.reduce_sum(layer1d*input_label, axis=[1, 2, 3, 4]) * 2
# adding a small value for numerical stability
dice_denominator = tf.reduce_sum(input_label, axis=[1, 2, 3, 4]) + tf.reduce_sum(layer1d, axis=[1, 2, 3, 4])+1e-6
dice = dice_numerator / dice_denominator
loss = 1 - tf.reduce_mean(dice)


# training op
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_iter = int(1e6)
n = 50  # 50 training image-label pairs
num_minibatch = int(n/size_minibatch)  # how many minibatches in each epoch
indices_train = [i for i in range(n)]
# data reader
path_to_data = './data'
DataFeeder = util.DataReader(path_to_data)


import random

# start the optimisation
for step in range(total_iter):

    # shuffle data every time start a new set of minibatches
    if step in range(0, total_iter, num_minibatch):
        random.shuffle(indices_train)

    # find out data indices for a minibatch
    minibatch_idx = step % num_minibatch  # minibatch index
    indices_mb = indices_train[minibatch_idx*size_minibatch:(minibatch_idx+1)*size_minibatch]
    trainFeed = {ph_image: DataFeeder.load_images_train(indices_mb),
                 ph_label: DataFeeder.load_labels_train(indices_mb)}

    # update the variables
    sess.run(train_op, feed_dict=trainFeed)

    # print training information
    if (step % 10) == 0:
        loss_train = sess.run(loss, feed_dict=trainFeed)
        print('Step %d: Loss=%f' % (step, loss_train))
    if (step % 100) == 0:
        dice_train = sess.run(dice, feed_dict=trainFeed)
        print('Individual training-Dice:')
        print(dice_train)

    # --- simple tests during training ---
    if (step % 500) == 0:
        num_test = int(30/size_minibatch)  # ignore the last few
        print('Individual test-Dice:')
        for test_idx in range(num_test):
            indices_test = list(range(test_idx*size_minibatch, (test_idx+1)*size_minibatch))
            testFeed = {ph_image: DataFeeder.load_images_test(indices_test)}
            layer1d_test = sess.run(layer1d, feed_dict=testFeed)
            # save the segmentation

sess.close()
