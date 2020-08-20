# cnn model
import numpy as np
from numpy import mean
from numpy import std
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 15, 2
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
    testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(n_timesteps,n_features, 1)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))

    tf.keras.utils.plot_model(model, show_shapes=False, to_file='figures/CNN_Conv2D.png')
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    confusionMatrix(model, testX, testy)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    Y_test = np.argmax(testy, axis=1)  # Convert one-hot to index
    y_pred = model.predict_classes(testX)
    print(classification_report(Y_test, y_pred))
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    print(trainX.shape, trainy.shape)
    #print(trainy)
    # repeat experiment
    scores = list()

    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    plt.savefig('figures/Confusion_Matrix_CNN_Conv2D.png', dpi=300)
    plt.show()

# run the experiment
run_experiment(5)