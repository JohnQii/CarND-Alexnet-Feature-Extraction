import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time
from datetime import timedelta
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
training_file = './train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
# TODO: Split data into training and validation sets.
X_train, y_train = train['features'], train['labels']
# TODO: Define placeholders and resize operation.
nb_classes = 43
rate = 0.00095

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fw = tf.Variable(tf.truncated_normal(shape,stddev=1e-2))
fb = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fw, fb)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, nb_classes)
keep_prob = tf.placeholder(tf.float32)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
# TODO: Train and evaluate the feature extraction model.
n_train = X_train.shape[0]
EPOCHS = 1
BATCH_SIZE = 100
with tf.Session() as sess:
    print("a ha, Training...")
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training with {} input images...".format(num_examples))
    print()
    for i in range(EPOCHS):
        start_time = time.time()
        # pre-process the image
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, n_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset: end], y_train[offset: end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        train_accuracy = evaluate(X_train, y_train)
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        time_diff = time.time() - start_time

        print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
        print()
    saver.save(sess, './Alexnet')
    print("Model save to Alexnet")