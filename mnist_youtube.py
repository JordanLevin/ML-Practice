import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

nodes_1 = 500
nodes_2 = 500
nodes_3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    layer_1 = {
        'weights': tf.Variable(tf.random_normal([784, nodes_1])),
        'biases': tf.Variable(tf.random_normal([ nodes_1 ]))
    }
    layer_2 = {
        'weights': tf.Variable(tf.random_normal([nodes_1, nodes_2])),
        'biases': tf.Variable(tf.random_normal([ nodes_2 ]))
    }
    layer_3 = {
        'weights': tf.Variable(tf.random_normal([nodes_2, nodes_3])),
        'biases': tf.Variable(tf.random_normal([ nodes_3 ]))
    }
    layer_output = {
        'weights': tf.Variable(tf.random_normal([nodes_3, n_classes])),
        'biases': tf.Variable(tf.random_normal([ n_classes ]))
    }

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, layer_1['weights']), layer_1['biases'])
    # use relu activation function
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, layer_3['weights']), layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, layer_output['weights']), layer_output['biases'])

    return output

def train_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #trains the model with training images
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('epoch ', epoch, 'completed out of ', epochs, 'loss: ', epoch_loss)

        #runs the model with the test data
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

if __name__ == "__main__":
    train_network(x)
