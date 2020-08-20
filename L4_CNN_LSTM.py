# cnn model
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os

#read the files
def getData(folder, name):
    data_path = folder + '/' + name
    data = np.loadtxt(data_path)
    f = open(data_path, "r")
    format = f.readline().replace('# Array shape: (', '').replace('\n', '').replace(')', '')
    format = np.array(format.split(', ')).astype(int)
    f.close()
    return data, format

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group
    # load all 9 files as a single array

    # total acceleration
    filenames = ['0_X.txt', '1_y.txt']

    # load input data
    X, format = getData(filepath, filenames[0])
    X = X.reshape(format[0], format[1], format[2])
    # load class output
    y, format = getData(filepath, filenames[1])
    y = y.reshape(format[0], format[1])
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train/', prefix + 'data/')
    # load all test
    testX, testy = load_dataset_group('test/', prefix + 'data/')
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = tf.keras.utils.to_categorical(trainy)
    testy = tf.keras.utils.to_categorical(testy)
    return trainX, trainy, testX, testy

#show confusion matrix
def confusionMatrix(model, testX, testy):
    y_pred = model.predict_classes(testX)
    y_test = np.argmax(testy, axis=1)  # Convert one-hot to index

    gestures = ['Martele', 'Staccato', 'Detache', 'Ricochet', 'Legato', 'Tremolo', 'Colle', 'Collegno']
    c_matrix = confusion_matrix(y_test, y_pred)
    norm_matrix = list()
    for row in c_matrix:
        m = np.sum(row)
        norm_matrix.append(np.true_divide(row, m))
    df_cm = pd.DataFrame(norm_matrix, index=[i for i in gestures],
                         columns=[i for i in gestures])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix')
    #sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, annot=True)

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, lSize):

    checkpoint_path = 'training_1' + '/'+ 'cp_' + str(lSize) + '.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=0)
    # define model
    batch_size = 32
    verbose, epochs = 0, 25 #best batch so far is 32
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 2, 48 #best option so far is 2 48
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    print('shape:', trainX.shape)
    # define model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(lSize, 3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(lSize, 3, activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D()))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=True, to_file='figures/CNN_LSTM.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cp_callback])
    # evaluate model
    confusionMatrix(model, testX, testy)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    Y_test = np.argmax(testy, axis=1)  # Convert one-hot to index
    y_pred = model.predict_classes(testX)
    print(classification_report(Y_test, y_pred))
    plt.savefig('figures/Confusion_Matrix_CNN_LSTM'+str(lSize)+'.png', dpi=300)
    # Display the model's architecture
    model.summary()
    return accuracy
# summarize scores
def summarize_results(scores, params, saveit = False):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param = %d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    if saveit:
        plt.boxplot(scores, labels=params)
        plt.savefig('figures/batches_CNN_LSTM_.png')
    plt.show()

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    print(trainX.shape, trainy.shape)
    #print(trainy)
    # repeat experiment
    final_scores = list()
    sizes = [32, 64, 128, 256, 512]

    for i in range(len(sizes)):
        scores = list()
        for r in range(repeats):
            score = evaluate_model(trainX, trainy, testX, testy, sizes[i])
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        final_scores.append(scores)
        tf.keras.backend.clear_session()
    # summarize results
    summarize_results(final_scores, sizes)
    #plt.show()

# run the experiment
run_experiment(10)