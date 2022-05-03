from Utilities import *

def gridSearch(X,Y):
    xgbclassifier = XGBClassifier(
                                max_depth=3, 
                                n_estimators=120,
                                learning_rate=0.1,
                                n_jobs=4,
                                tree_method="hist",
                                eval_metric="error",
                                scale_pos_weight=sum_wbkg/sum_wsig,
                                objective='binary:logistic',
                                missing=-999.0,
                                use_label_encoder=False) 
    param_dist = {"n_estimators": sp_randint(1, 200),
              "max_depth": sp_randint(1,20),
              "learning_rate": sp_uniform(0.1,1.0),
              "gamma": sp_uniform(0,10),
              "min_child_weight": sp_uniform(0,10),
             }


    # run random search
    n_iter_search = 50
    random_search = RandomizedSearchCV(xgbclassifier, 
                                    param_distributions=param_dist, 
                                    n_iter=n_iter_search,
                                    verbose=2,
                                    n_jobs=4,
                                    cv=3,
                                    scoring='roc_auc')
    random_search.fit(X, Y)

    return random_search.best_estimator_
   

preformGS = ask("Preform gird search? [y/n]")

start_time = timer(None)
print(" ")
print(" ___________*XGBoost*___________")
print("|      Reading CSV-file...      |")
data = pd.read_csv("../data/MC_data_gammel.csv")
nrEvent, nrFeatures = data.shape
print(f"| Features: {nrFeatures}, Events: {nrEvent:.2e}|")

Y =  np.array(data.label)
X = np.array(data.drop(["label"],axis=1))
w_train = np.array(data.weight)

nrS = np.sum(Y == 1)
nrB = sum(Y == 0)
print(f"|   B: {nrB:.2e}   S: {nrS:.2e}   |")

X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, W_test= splitData(X,Y,0.2)

# Weights
wbkg_t = [W_test[i] for i in range(len(Y_test)) if Y_test[i] == 0.0 ]

wsig = [W_train[i] for i in range(len(Y_train)) if Y_train[i] == 1.0 ]
wbkg = [W_train[i] for i in range(len(Y_train)) if Y_train[i] == 0.0 ]

wsig_v = [W_val[i] for i in range(len(Y_val)) if Y_val[i] == 1.0 ]
wbkg_v = [W_val[i] for i in range(len(Y_val)) if Y_val[i] == 0.0 ]


sum_wsig = sum( wsig )
sum_wbkg = sum( wbkg )


if preformGS:
    global XSize 
    XSize = len(X[0])
    xgb = gridSearch(X_train,Y_train)
    print(xgb.get_xgb_params())

    
else:
    xgb = XGBClassifier(
                        max_depth=3, 
                        n_estimators=120,
                        learning_rate=0.1,
                        n_jobs=4,
                        tree_method="hist",
                        objective='binary:logistic',
                        scale_pos_weight=sum_wbkg/sum_wsig,
                        missing=-999.0,
                        use_label_encoder=False,
                        eval_metric="error") 

    print("|          Training...          |")
    xgb.fit(X_train,Y_train)

    print("|        Finished traing        |")
    print(" ------------------------------- ")
timer(start_time)

"""
TRAINING DATA
"""
y_pred = xgb.predict_proba( X_train ) 
y_b = y_pred[:,1][Y_train==0]
y_s = y_pred[:,1][Y_train==1]
name = "../figures/XGB/train.pdf"
title =  "XGB output, MC-data, training data"
plotHistoBS(y_b, y_s, wbkg, wsig, name, title,  nrBins = 15)

title = "ROC for XGB on MC-dataset (training)"

plotRoc(Y_train, y_pred, title)

"""
VALIDATION DATA
"""
y_pred = xgb.predict_proba( X_val ) 
y_b = y_pred[:,1][Y_val==0]
y_s = y_pred[:,1][Y_val==1]
name = "../figures/XGB/val.pdf"
title =  "XGB output, MC-data, validation data"
plotHistoBS(y_b, y_s, wbkg_v, wsig_v, name, title,  nrBins = 15)

title = "ROC for XGB on MC-dataset (valdiation)"
plotRoc(Y_val, y_pred, title)


"""
TEST DATA
"""
y_b = xgb.predict_proba( X_test )[:,1]

name = "../figures/XGB/test.pdf"
title =  "XGB output, MC-data, test data"
plotHistoB(y_b, wbkg_t, name, title,  nrBins = 15)



if  ask("Do you want to save model? (y/n) "):
    dirname = os.getcwd()
    filename = os.path.join(dirname, f"../models/XGB/hyperModel2.joblib")
    dump(xgb, filename)
    print("Model saved")
else:
    print("Model not saved")
