from Utilities import *
import tensorflow as tf
from tensorflow.keras import optimizers
import keras_tuner as kt
tf.random.set_seed(1)

preformGS = ask("Preform gird search? [y/n]")

def gridSearch(X,Y):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="GridSearches",
        project_name="NN",
        overwrite=True,
    )

    tuner.search(X, Y, epochs=10,batch_size = 25000, validation_split = 0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number nodes in start, first and second layer is {best_hps.get('num_of_neurons0')}, {best_hps.get('num_of_neurons1')} and \
    {best_hps.get('num_of_neurons2')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """
    )
    if  ask("Do you want to save model? (y/n) "):
        tuner.hypermodel.build(best_hps).save("../models/NeuralNetwork/hyperModel.h5")
        print("Model saved")
    else:
        print("Model not saved")


def model_builder(hp):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons0", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                kernel_initializer = tf.keras.initializers.HeUniform(seed = 1),
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
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3])
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

scaler = StandardScaler()
X = scaler.fit_transform(X)

nrS = np.sum(Y == 1)
nrB = sum(Y == 0)
print(f"|     B: {nrB:.2e}   S: {nrS:.2e}    |")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=2)
w_train, w_val, Y_fill, Y_fill = train_test_split(w_train, Y, test_size=0.2, random_state=2)


wsig_v = [w_val[i] for i in range(len(Y_val)) if Y_val[i] == 1.0 ]
wbkg_v = [w_val[i] for i in range(len(Y_val)) if Y_val[i] == 0.0 ]


wsig = [w_train[i] for i in range(len(Y_train)) if Y_train[i] == 1.0 ]
wbkg = [w_train[i] for i in range(len(Y_train)) if Y_train[i] == 0.0 ]

sum_wsig = sum( wsig )
sum_wbkg = sum( wbkg )

if preformGS:
    global XSize 
    XSize = len(X[0])
    gridSearch(X_train,Y_train)
else:
    #model = tf.keras.models.load_model(f"../models/NeuralNetwork/hyperModel.h5")
    initializer = tf.keras.initializers.HeUniform(seed = 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=40,
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                input_shape=(len(X_train[0]),),
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                units=50,
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                units=30,
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    learning_rate = 0.005
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    print("|            Training...           |")

    model.fit(X_train,Y_train, epochs = 100, batch_size = 25000) 

    print("|          Finished traing         |")
    print(" ---------------------------------- ")
    y_pred_prob = model.predict( X_val ) 

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    n, bins, patches = plt.hist(y_pred_prob[Y_val==0], bins = np.linspace(0,1.,15), facecolor='blue', alpha=0.2,label="Background",density=True)# weights = wbkg_v)#, density=True)
    n, bins, patches = plt.hist(y_pred_prob[Y_val==1], bins = np.linspace(0,1.,15), facecolor='red', alpha=0.2, label="Signal",density=True)# weights = wsig_v,density=True)#, density=True)
    plt.xlabel('XGBoost output',fontsize=14)
    plt.ylabel('Events',fontsize=14)
    plt.title('Neural network output, MC-data, validation data',fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.ylim(bottom=1e-3)  # adjust the bottom leaving top unchanged
    plt.xlim([0.01,1.01])
    if ask("Save Image? [y/n]"):
        plt.savefig("../figures/NeuralNetwork/OutDistExtraCuts_NI.pdf", bbox_inches="tight")
    plt.show()

    fpr, tpr, thresholds = roc_curve(Y_val,y_pred_prob, pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for xgBoost on MC-dataset')
    plt.legend(loc="lower right")
    plt.show()

