from Utilities import *
import tensorflow as tf
from tensorflow.keras import optimizers
import keras_tuner as kt


state = True
while state == True:
        answ = input("Preform gird search? [y/n]")
        if answ == "y":
            preformGS = True
            state = False
        elif answ == "n":
            preformGS = False
            state = False

def gridSearch(X,Y):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="GridSearches",
        project_name="NN",
        overwrite=True,
    )

    tuner.search(X, Y, epochs=50, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number nodes in start, first, second layer is {best_hps.get('num_of_neurons0')}, {best_hps.get('num_of_neurons1')} and \
    {best_hps.get('num_of_neurons2')} and third layer {best_hps.get('num_of_neurons3')} the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """
    )

    state = True
    while state == True:
        answ = input("Do you want to save model? (y/n) ")
        if answ == "y":
            name = input("name: ")
            tuner.hypermodel.build(best_hps).save(f"../models/NeuralNetwork/model_{name}.h5")
            state = False
            print("Model saved")
        elif answ == "n":
            state = False
            print("Model not saved")


def model_builder(hp):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons0", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                input_shape=(XSize,),
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons1", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons2", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons3", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    hp_learning_rate = hp.Choice("learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3])
    optimizer = optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

start_time = timer(None)
print(" ")
print(" _________*Neural Network*_________")
print("|        Reading CSV-file...       |")
data = pd.read_csv("../data/MC_data.csv")

nrEvent, nrFeatures = data.shape
print(f"|  Features: {nrFeatures}, Events: {nrEvent:.2e}  |")

Y =  np.array(data.label)
X = np.array(data.drop(["label", "weight"],axis=1))
w_train = np.array(data.weight)

nrS = np.sum(Y == 1)
nrB = sum(Y == 0)
print(f"|     B: {nrB:.2e}   S: {nrS:.2e}    |")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=2)
if preformGS:
    global XSize 
    XSize = len(X[0])
    gridSearch(X_train,Y_train)
else:
    model = tf.keras.models.load_model(f"../models/NeuralNetwork/hyperModel.h5")
    print("|            Training...           |")

    model.fit(X_train,Y_train) 

    print("|          Finished traing         |")
    print(" ---------------------------------- ")

