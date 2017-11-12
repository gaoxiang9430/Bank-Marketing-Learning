from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import csv
from sklearn.cross_validation import train_test_split

model_name = 'model'
#model_path_model_1 = '/home/sandareka/Tensorflow/Data_Mining_Project_1' #'F:/Data Mining Project/Models' #path to save the model
#model_path_model_2 = '/home/sandareka/Tensorflow/Data_Mining_Project_2'

model_path_model_1 = '/home/sandareka/Tensorflow/Data_Mining_Project_3' #'F:/Data Mining Project/Models' #path to save the model
model_path_model_2 = '/home/sandareka/Tensorflow/Data_Mining_Project_4'


is_GPU = True

if is_GPU == True:
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"


def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    print("loading training and validation data")
    trainX =  np.delete(np.genfromtxt("data_nn/pre_processed_training_data_11_12_6.csv", delimiter=","), 0, axis=0)
    trainY = np.delete(np.genfromtxt("data_nn/training_labels_11_12_6.csv", delimiter=","), 0, axis=0)

    #Loading data for validation
    validX =  np.delete(np.genfromtxt("data_nn/pre_processed_validation_data_11_12_6.csv", delimiter=","), 0, axis=0)
    validY = np.delete(np.genfromtxt("data_nn/validation_labels_11_12_6.csv", delimiter=","), 0, axis=0)

    trainX_, validationX, trainY_, validationY = train_test_split(validX, validY, test_size=0.1, random_state=100)

    print("loading testing data")
    testX = np.delete(np.genfromtxt("data_nn/pre_processed_testing_data_11_12_6.csv", delimiter=","), 0, axis=0)
    return trainX,trainY,validationX,validationY,testX

#Given the label returns one hot representation of that lable
def convert_to_one_hot_representation(data_set):
    new_data_set = np.zeros(shape=(len(data_set), 2))
    for i in range(len(data_set)):
        new_data_set[i, int(data_set[i])] = 1

    return new_data_set

def write_output_file(output_file, output):
    with open(output_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for r in range(len(output)):
            output_Text = [r,output[r]]
            writer.writerow(output_Text)

def draw_graphs(mean_loss_for_epoch,mcc_scores,model_name):

    if len(mean_loss_for_epoch) >0:
        plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.plot(mean_loss_for_epoch, '-o')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        plt.savefig('training_loss_'+model_name+'png')
        plt.clf()

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.plot(mcc_scores, '-o')
    plt.xlabel('epoch')
    plt.ylabel('validation mcc score')
    plt.savefig('validation_mcc_'+model_name+'.png')
    plt.clf()

def draw_graphs_mcc(model1,model2,ensemble_model):

    """
    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    """

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.plot(model1, '-o')
    plt.plot(model2, '-o')
    plt.plot(ensemble_model, '-o')
    plt.xlabel('epoch')
    plt.ylabel('validation mcc score')
    plt.legend(['Model 1', 'Model 2','Ensemble model'], loc='upper left')
    plt.savefig('validation_mcc_comparison.png')
    plt.clf()

trainX,train_label,validationX,validation_label,testX = import_data()

trainY = convert_to_one_hot_representation(train_label)
validationY = convert_to_one_hot_representation(validation_label)

# Parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 50
display_step = 1
positive_class_frequency = 0.12
negative_class_frequency = 0.88

#200 x 4, 0.0001, 1000, 57.49


n_input = trainX.shape[1] # Number of features
n_classes = 2 #Binary classification

with tf.device(device_name):
    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    given_labels = tf.placeholder("float", [None])
    dropout_keep_prob = tf.placeholder(tf.float32)

    initializer = tf.contrib.layers.xavier_initializer()

    # Store layers weight & bias


    # Create model
    def multilayer_perceptron_1(x):
        # Network Parameters
        n_hidden_1 = 256  # 2048#512#128#2500#256 # 1st layer number of neurons
        n_hidden_2 = 256  # 1024#256#64#2000#256 # 2nd layer number of neurons
        n_hidden_3 = 256  # 512#1500#256 # 3rd layer number of neurons
        n_hidden_4 = 256  # 512#200 # 4th layer number of neurons
        n_hidden_5 = 256
        n_hidden_6 = 256
        n_hidden_7 = 256

        weights = {
            'h1': tf.Variable(initializer([n_input, n_hidden_1])),
            'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(initializer([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(initializer([n_hidden_3, n_hidden_4])),
            'h5': tf.Variable(initializer([n_hidden_4, n_hidden_5])),
            'h6': tf.Variable(initializer([n_hidden_5, n_hidden_6])),
            'h7': tf.Variable(initializer([n_hidden_6, n_hidden_7])),
            'out': tf.Variable(initializer([n_hidden_7, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'b3': tf.Variable(tf.zeros([n_hidden_3])),
            'b4': tf.Variable(tf.zeros([n_hidden_4])),
            'b5': tf.Variable(tf.zeros([n_hidden_5])),
            'b6': tf.Variable(tf.zeros([n_hidden_6])),
            'b7': tf.Variable(tf.zeros([n_hidden_7])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), dropout_keep_prob)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), dropout_keep_prob)
        # Hidden fully connected layer with 256 neurons
        layer_3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), dropout_keep_prob)

        layer_4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])), dropout_keep_prob)

        layer_5 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])), dropout_keep_prob)

        layer_6 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])), dropout_keep_prob)

        layer_7 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])), dropout_keep_prob)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.tanh(tf.add(tf.matmul(layer_7, weights['out']),biases['out']))
        return out_layer

    def multilayer_perceptron_2(x):
        # Network Parameters
        n_hidden_1 = 128  # 2048#512#128#2500#256 # 1st layer number of neurons
        n_hidden_2 = 128  # 1024#256#64#2000#256 # 2nd layer number of neurons
        n_hidden_3 = 128  # 512#1500#256 # 3rd layer number of neurons
        n_hidden_4 = 128  # 512#200 # 4th layer number of neurons
        n_hidden_5 = 128
        n_hidden_6 = 128

        weights = {
            'h1': tf.Variable(initializer([n_input, n_hidden_1])),
            'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(initializer([n_hidden_2, n_hidden_3])),
            'h4': tf.Variable(initializer([n_hidden_3, n_hidden_4])),
            'h5': tf.Variable(initializer([n_hidden_4, n_hidden_5])),
            'h6': tf.Variable(initializer([n_hidden_5, n_hidden_6])),
            'out': tf.Variable(initializer([n_hidden_6, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'b3': tf.Variable(tf.zeros([n_hidden_3])),
            'b4': tf.Variable(tf.zeros([n_hidden_4])),
            'b5': tf.Variable(tf.zeros([n_hidden_5])),
            'b6': tf.Variable(tf.zeros([n_hidden_6])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), dropout_keep_prob)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), dropout_keep_prob)
        # Hidden fully connected layer with 256 neurons
        layer_3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), dropout_keep_prob)

        layer_4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])), dropout_keep_prob)

        layer_5 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])), dropout_keep_prob)

        layer_6 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])), dropout_keep_prob)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.tanh(tf.add(tf.matmul(layer_6, weights['out']),biases['out']))
        return out_layer




    # Construct model
    logits_model_1 = multilayer_perceptron_1(X)

    logits_model_2 = multilayer_perceptron_2(X)

    softmaxed_logits_model_1 = tf.nn.softmax(logits=logits_model_1, dim=1)
    predicted_results_model_1 = tf.argmax(softmaxed_logits_model_1, 1)

    softmaxed_logits_model_2 = tf.nn.softmax(logits=logits_model_2, dim=1)
    predicted_results_model_2 = tf.argmax(softmaxed_logits_model_2, 1)

    softmaxed_logits_ensemble_model = tf.reduce_mean([softmaxed_logits_model_1,softmaxed_logits_model_2],0)
    predicted_results_ensemble_model = tf.argmax(softmaxed_logits_ensemble_model, 1)

    auc_score_1 = tf.metrics.auc(given_labels, predicted_results_model_1)
    auc_score_2 = tf.metrics.auc(given_labels, predicted_results_model_2)
    auc_score_ensemble = tf.metrics.auc(given_labels, predicted_results_ensemble_model)

    #predicted_results = tf.greater(softmaxed_logits, 1)

    # Define loss and optimizer

    loss_op_model_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_model_1, labels=Y))
    loss_op_model_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_model_2, labels=Y))


    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, int(trainX.shape[0] / batch_size),0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op_model_1 = optimizer.minimize(loss_op_model_1)
    train_op_model_2 = optimizer.minimize(loss_op_model_2)


    # Initializing the variables
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    saver_model_1 = tf.train.Saver(max_to_keep=100)
    losses_model_1 = []
    mean_loss_for_epoch_model_1 = []
    mcc_scores_model_1 = []
    best_mcc_model_1 = 0
    auc_scores_model_1 = []
    best_auc_model_1 = 0

    saver_model_2 = tf.train.Saver(max_to_keep=100)
    losses_model_2 = []
    mean_loss_for_epoch_model_2 = []
    mcc_scores_model_2 = []
    best_mcc_model_2 = 0
    auc_scores_model_2 = []
    best_auc_model_2 = 0

    mcc_scores_model_ensemble = []
    best_mcc_model_ensemble = 0
    auc_scores_ensemble = []
    best_auc_ensemble = 0

    global_step = tf.Variable(0, trainable=False)
    sess.run(init)


    # Training cycle
    for epoch in range(training_epochs):
        index = (np.arange(len(trainX)).astype(int))
        np.random.shuffle(index)
        for start, end in zip(range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
            batch_x = trainX[index[start:end]]
            batch_y = trainY[index[start:end]]
            batch_lable = train_label[index[start:end]]
            dropout_keep_prob_input = 1.0#0.5
            # Run optimization op (backprop) and cost op (to get loss value)
            _, loss_value_1 = sess.run([train_op_model_1, loss_op_model_1], feed_dict={X: batch_x,
                                                            Y: batch_y, dropout_keep_prob:dropout_keep_prob_input,given_labels:batch_lable})

            _, loss_value_2 = sess.run([train_op_model_2, loss_op_model_2], feed_dict={X: batch_x,
                                                                                       Y: batch_y,
                                                                                       dropout_keep_prob: dropout_keep_prob_input,
                                                                                       given_labels:batch_lable})
            # Print average loss for the batch
            print("Model 1 - Current Cost: ", loss_value_1, "\t Epoch {}/{}".format(epoch, training_epochs),
                  "\t Iter {}/{}".format(start, len(trainX)))
            losses_model_1.append(loss_value_1)

            print("Model 2 - Current Cost: ", loss_value_2, "\t Epoch {}/{}".format(epoch, training_epochs),
                "\t Iter {}/{}".format(start, len(trainX)))
            losses_model_2.append(loss_value_2)

        #Average loss for the epoch
        mean_loss_for_epoch_model_1.append(np.mean(losses_model_1))
        mean_loss_for_epoch_model_2.append(np.mean(losses_model_2))

        print("Model 1-Mean loss for the epoch", mean_loss_for_epoch_model_1[epoch])
        print("Model 2-Mean loss for the epoch", mean_loss_for_epoch_model_2[epoch])

        #Validation-------------------------------------
        #Predict class for the validation set and calculate MCC score
        dropout_keep_prob_input = 1.0

        returned_predicted_results_model_1,returned_predicted_results_model_2,returned_predicted_results_ensemble, returned_auc_score_1,returned_auc_score_2, returned_auc_score_ensemble = sess.run([predicted_results_model_1,predicted_results_model_2,predicted_results_ensemble_model,auc_score_1,auc_score_2,auc_score_ensemble], feed_dict={X: validationX, Y: validationY, dropout_keep_prob:dropout_keep_prob_input,given_labels:validation_label})

        mcc_score_model_1 = matthews_corrcoef(validation_label, returned_predicted_results_model_1)
        print("Model 1 - MCC Score: ", mcc_score_model_1)
        mcc_scores_model_1.append(mcc_score_model_1)


        if best_auc_model_1 < returned_auc_score_1[0]:
            best_auc_model_1 = returned_auc_score_1[0]

        #Save model with the best validation results
        if best_mcc_model_1< mcc_score_model_1:
            best_mcc_model_1 = mcc_score_model_1
            print("Model 1 - Saving the model from epoch: ", epoch)
            saver_model_1.save(sess, os.path.join(model_path_model_1, model_name), global_step=epoch)


        mcc_score_model_2 = matthews_corrcoef(validation_label, returned_predicted_results_model_2)
        print("Model 2 - MCC Score: ", mcc_score_model_2)
        mcc_scores_model_2.append(mcc_score_model_2)


        if best_auc_model_2 < returned_auc_score_2[0]:
            best_auc_model_2 = returned_auc_score_2[0]

        #Save model with the best validation results
        if best_mcc_model_2< mcc_score_model_2:
            best_mcc_model_2 = mcc_score_model_2
            print("Model 2 - Saving the model from epoch: ", epoch)
            saver_model_2.save(sess, os.path.join(model_path_model_2, model_name), global_step=epoch)

        mcc_score_model_ensemble = matthews_corrcoef(validation_label, returned_predicted_results_ensemble)
        print("Ensemble model - MCC Score: ", mcc_score_model_ensemble)
        mcc_scores_model_ensemble.append(mcc_score_model_ensemble)

        if best_mcc_model_ensemble < mcc_score_model_ensemble:
            best_mcc_model_ensemble = mcc_score_model_ensemble

        #Save model with the best validation results
        if best_auc_ensemble< returned_auc_score_ensemble [0]:
            best_auc_ensemble = returned_auc_score_ensemble[0]

    draw_graphs_mcc(mcc_scores_model_1, mcc_scores_model_2, mcc_scores_model_ensemble)

    draw_graphs(mean_loss_for_epoch_model_1,mcc_scores_model_1,"model_1")
    draw_graphs(mean_loss_for_epoch_model_2, mcc_scores_model_2, "model_2")
    draw_graphs([], mcc_scores_model_ensemble, "model_ensemble")




    print("Optimization Finished!")
    print("Model 1 Best MCC Score Achieved", best_mcc_model_1)
    print("Model2 Best AUC", best_auc_model_1)
    print("Model 2 Best MCC Score Achieved", best_mcc_model_2)
    print("Model2 Best AUC", best_auc_model_2)
    print("Ensemble Model Best MCC Score Achieved", best_mcc_model_ensemble)
    print("Ensemble Model Best AUC", best_auc_ensemble)

"""
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Test model
    logits_model_1 = multilayer_perceptron_1(X)
    saver = tf.train.Saver()
    saved_path = tf.train.latest_checkpoint(model_path_model_1)
    saver.restore(sess, saved_path)
    dropout_keep_prob_input = 1.0
    softmaxed_logits_model_1 = tf.nn.softmax(logits=logits_model_1, dim=1)

    #predicted_results_model_1 = tf.argmax(softmaxed_logits_model_1, 1)
    #returned_predicted_results = sess.run(predicted_results_model_1, feed_dict={X: testX, dropout_keep_prob: dropout_keep_prob_input})
    test_results_model_1 = []
    n_test_input = testX.shape[0]
    
    #for i in range (n_test_input):
    returned_softmaxed_logits_model_1 = sess.run(softmaxed_logits_model_1,feed_dict={X: testX, dropout_keep_prob: dropout_keep_prob_input})


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # Test model
    logits_model_2 = multilayer_perceptron_2(X)
    saver_model_2 = tf.train.Saver()
    saved_path_model_2 = tf.train.latest_checkpoint(model_path_model_2)
    saver_model_2.restore(session, saved_path_model_2)
    dropout_keep_prob_input = 1.0
    softmaxed_logits_model_2 = tf.nn.softmax(logits=logits_model_2, dim=1)

    # predicted_results_model_1 = tf.argmax(softmaxed_logits_model_1, 1)
    # returned_predicted_results = sess.run(predicted_results_model_1, feed_dict={X: testX, dropout_keep_prob: dropout_keep_prob_input})

    returned_softmaxed_logits_model_2 = session.run(softmaxed_logits_model_2,feed_dict={X: testX, dropout_keep_prob: dropout_keep_prob_input})

softmaxed_logits_ensemble_model = np.mean([returned_softmaxed_logits_model_1, returned_softmaxed_logits_model_2], 0)
predicted_results_ensemble_model = np.argmax(softmaxed_logits_ensemble_model, 1)
write_output_file("results_11_12_4.csv",predicted_results_ensemble_model)
"""