from xgboost import XGBClassifier
from Utilities import *
start_time = timer(None)
print(" ")
print(" ____________XGBoost____________")
print("|      Reading CSV-file...      |")
data = pd.read_csv("../data/MC_data.csv")
print("|        File -> Read           |")

Y =  np.array(data.label)
X = np.array(data.drop(["label", "weight"],axis=1))
w_train = np.array(data.weight)


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

#wsig_v = [w_train[i] for i in range(len(Y_val)) if Y_val[i] == 1.0 ]
#wbkg_v = [w_train[i] for i in range(len(Y_val)) if Y_val[i] == 0.0 ]

wsig = [w_train[i] for i in range(len(Y_train)) if Y_train[i] == 1.0 ]
wbkg = [w_train[i] for i in range(len(Y_train)) if Y_train[i] == 0.0 ]
sum_wsig = sum( wsig )
sum_wbkg = sum( wbkg )

xgb = XGBClassifier(
                    max_depth=3, 
                    n_estimators=120,
                    learning_rate=0.1,
                    n_jobs=4,
                    tree_method="hist",
                    objective='binary:logistic',
                    missing=-999.0,
                    use_label_encoder=False,
                    eval_metric="error") 
                    
print("|          Training...          |")

xgb.fit(X_train,Y_train) 

print("|        Finished traing        |")
print(" ------------------------------- ")
timer(start_time)

y_pred_prob = xgb.predict_proba( X_val ) # The BDT outputs for each eventb.predict(X_val)

n, bins, patches = plt.hist(y_pred_prob[:,1][Y_val==0], 100, facecolor='blue', alpha=0.2,label="Background", weights = wbkg)#, density=True)
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_val==1], 100, facecolor='red', alpha=0.2, label="Signal", weights = wsig)#, density=True)

plt.xlabel('XGBoost output')
plt.ylabel('Events')
plt.title('XGBoost output, HiggsML dataset, validation data')
plt.grid(True)
plt.legend()
plt.yscale('log')
plt.show()

fpr, tpr, thresholds = roc_curve(Y_val,y_pred_prob[:,1], pos_label=1)
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
