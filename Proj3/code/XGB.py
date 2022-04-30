from xgboost import XGBClassifier
from Utilities import *
start_time = timer(None)
print(" ")
print(" ___________*XGBoost*___________")
print("|      Reading CSV-file...      |")
data = pd.read_csv("../data/MC_data.csv")
nrEvent, nrFeatures = data.shape
print(f"| Features: {nrFeatures}, Events: {nrEvent:.2e}|")


Y =  np.array(data.label)
X = np.array(data.drop(["label"],axis=1))
w_train = np.array(data.weight)

nrS = np.sum(Y == 1)
nrB = sum(Y == 0)
print(f"|   B: {nrB:.2e}   S: {nrS:.2e}   |")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=2)

w_train = X_train[:,-2]
w_val = X_val[:,-2]

X_train, X_val = X_train[:,:-1], X_val[:,:-1]

wsig_v = [w_val[i] for i in range(len(Y_val)) if Y_val[i] == 1.0 ]
wbkg_v = [w_val[i] for i in range(len(Y_val)) if Y_val[i] == 0.0 ]


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
                    scale_pos_weight=sum_wbkg/sum_wsig,
                    missing=-999.0,
                    use_label_encoder=False,
                    eval_metric="error") 

print("|          Training...          |")

xgb.fit(X_train,Y_train) 

print("|        Finished traing        |")
print(" ------------------------------- ")
timer(start_time)

y_pred_prob = xgb.predict_proba( X_val ) 

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_val==0], 10, facecolor='blue', alpha=0.2,label="Background", weights = wbkg_v,density=True)#, density=True)
n, bins, patches = plt.hist(y_pred_prob[:,1][Y_val==1], 10, facecolor='red', alpha=0.2, label="Signal", weights = wsig_v,density=True)#, density=True)
plt.xlabel('XGBoost output',fontsize=14)
plt.ylabel('Events',fontsize=14)
plt.title('XGBoost output, MC-data, validation data',fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.yscale('log')
plt.savefig("../figures/XGB/OutDistExtraCuts.pdf", bbox_inches="tight")
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
