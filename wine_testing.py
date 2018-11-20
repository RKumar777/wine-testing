
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#reading csv and splitting in test and train for further evaluation

data=pd.read_csv("wine.csv")
data=data.replace(to_replace='high',value=1)
data=data.replace(to_replace='low',value=0)
tra,tes = train_test_split(data, test_size=0.2)
tra_x=tra.iloc[:,0:tra.shape[1]-1]
tra_y=tra.iloc[:,tra.shape[1]-1]
tes_x=tes.iloc[:,0:tes.shape[1]-1]
tes_y=tes.iloc[:,tes.shape[1]-1]
kfolds=7
cutter=0.5

def get_pred_logreg(train,test):
    train_x=train.iloc[:,0:train.shape[1]-1]
    train_y=train.iloc[:,train.shape[1]-1]
    test_x=test.iloc[:,0:test.shape[1]-1]
    test_y=test.iloc[:,test.shape[1]-1]
    logreg=linear_model.LogisticRegression()
    logreg.fit(train_x, train_y)
    pred=logreg.predict_proba(test_x)[:,1]
    d = {'predicted_output_probs': pred, 'true_output': test_y}
    prix = pd.DataFrame(data=d)
    return prix

def get_pred_svm(train,test):
    train_x=train.iloc[:,0:train.shape[1]-1]
    train_y=train.iloc[:,train.shape[1]-1]
    test_x=test.iloc[:,0:test.shape[1]-1]
    test_y=test.iloc[:,test.shape[1]-1]
    predsvm=svm.SVC(probability=True)
    predsvm.fit(train_x,train_y)
    pred=predsvm.predict(test_x)
    d = {'predicted_output_probs': pred, 'true_output': test_y}
    prix = pd.DataFrame(data=d)
    return prix

def get_pred_nb(train,test):
    train_x=train.iloc[:,0:train.shape[1]-1]
    train_y=train.iloc[:,train.shape[1]-1]
    test_x=test.iloc[:,0:test.shape[1]-1]
    test_y=test.iloc[:,test.shape[1]-1]
    prednb=GaussianNB()
    prednb.fit(train_x,train_y)
    pred=prednb.predict_proba(test_x)[:,1]
    d = {'predicted_output_probs': pred, 'true_output': test_y}
    prix = pd.DataFrame(data=d)
    return prix

def get_pred_knn (train,test,k):
    train_x=train.iloc[:,0:train.shape[1]-1]
    train_y=train.iloc[:,train.shape[1]-1]
    test_x=test.iloc[:,0:test.shape[1]-1]
    test_y=test.iloc[:,test.shape[1]-1]
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y)
    pred=neigh.predict_proba(test_x)[:,1]
    d = {'predicted_output_probs': pred, 'true_output': test_y}
    prix = pd.DataFrame(data=d)
    return prix

# print(get_pred_logreg(tra,tes))
print(get_pred_svm(tra,tes))
# print(get_pred_nb(tra,tes))  
# print(get_pred_knn(tra,tes,1))

# Building the do_cv_class function

def do_cv_class(df, num_folds, model_name):
#     Randomize sample
    df=df.sample(df.shape[0])
    modex='get_pred_'
    model_name=modex+model_name
    if(model_name[-1]=='n'):
        grr=model_name.find('n')
        k=int(model_name[9:grr])
        model_name=model_name.replace(str(k),'k')
    n,m=df.shape
    remain=np.remainder(n,num_folds)
    times=int(n/num_folds)
    res=pd.DataFrame()
    adder=[]
    folder=0
    for j in range(num_folds):
        s1=times*(j)
        s2=times*(j+1)
        dftest=df.iloc[s1:s2]
        dftrain=df.drop(df.index[s1:s2])
        folder=folder+1
        if(model_name[-1]=='n'):
            reshu=eval(model_name)(dftrain,dftest,k)
            saz=dftest.shape[0]
            adder=np.ones(saz)
            adder=adder*folder
            reshu.insert(2,'fold',adder)
            res=res.append(reshu)
            
        else:
            reshu=eval(model_name)(dftrain,dftest)
            saz=dftest.shape[0]
            adder=np.ones(saz)
            adder=adder*folder
            reshu.insert(2,'fold',adder)
            
            res=res.append(reshu)
    return res
    
actuals=data.iloc[:,-1]

# print(do_cv_class(data,7,'nb'))  
# print(do_cv_class(data,7,'1nn')) 
# print(do_cv_class(data,7,'svm'))
# print(do_cv_class(data,7,'logreg')) 


# predics=do_cv_class(data,10,'nb')+do_cv_class(data,10,'5nn')+do_cv_class(data,10,'svm')+do_cv_class(data,10,'logreg')
# predics=predics.iloc[:,0]
# predics=pd.DataFrame(predics)
# predics.insert(1,'true_output',actuals)
# predics.iloc[:,0]=predics.iloc[:,0]/4

# Building the get_metrics function

def get_metrics(predicted,coff):
    tp=0
    fp=0
    tn=0
    fn=0
    gota=[1 if x > coff else 0 for x in predicted.iloc[:,0]]
    totp=np.sum([1 if x==1 else 0 for x in predicted.iloc[:,1]])
    totn=np.sum([1 if x==0 else 0 for x in predicted.iloc[:,1]])
    predicted.iloc[:,0]=gota
    # Getting values to calculate the function outputs
    for j in range(predicted.shape[0]):
        if(predicted.iloc[j,0]==1 and predicted.iloc[j,1]==1):
            tp=tp+1
        if(predicted.iloc[j,0]==1 and predicted.iloc[j,1]==0):
            fp=fp+1
        if(predicted.iloc[j,0]==0 and predicted.iloc[j,1]==0):
            tn=tn+1
        if(predicted.iloc[j,0]==0 and predicted.iloc[j,1]==1):
            fn=fn+1
    tpr=float(tp/totp)
    fpr=float(fp/totn)
    acc=float((tp+tn)/(totp+totn))
    precision=float(tp/(tp+fp))
    recall=tpr
    return tpr, fpr, acc, precision, recall
# print(get_metrics(do_cv_class(data,7,'nb'),0.5))
compra=[]
nums=[]

dps=data.shape[0]-int(data.shape[0]/kfolds)
for j in range(dps):
    guddu=[]
    guddu=get_metrics(do_cv_class(data,kfolds,str(j+1)+'nn'),cutter)
    compra.append(guddu[2])
print("Knn with k = {} gives best generalization".format(compra.index(max(compra))+1))
print("The accuracy provided with k={} is {}".format(compra.index(max(compra))+1,max(compra)))

# plotting accuracy values to check for overfitting/underfitting

import matplotlib.pyplot as plt
compra2=np.ones(len(compra))-compra
rannums=np.arange(dps)+np.ones(dps)
plt.plot(rannums,compra)
plt.xlabel('K-folds')
plt.ylabel('Accuracy')
plt.title("Accuracy vs k-folds")
plt.savefig('Acc.png')
plt.show()

# getting accuracy values for all the classifiers and the default one

atr=['nb','logreg','svm']
hst=0
qst=[]
y=0
ctrr=''
for q in atr:
    qst.append(get_metrics(do_cv_class(data,kfolds,q),cutter)[2]) 
    if(qst[y]>hst):
        hst=max(qst)
        ctrr=q
    y=y+1
print(qst)
print("The parametric model giving highest accuracy is :{}".format(ctrr))
print("The accuracy of {} is {}".format(ctrr,hst))
# default classifier
# print(data.iloc[:,3].value_counts().index(max(data.iloc[:,3].value_counts())))
grew=(data.iloc[:,3].value_counts())

defacc=max(grew)/sum(grew)
print("Accuracy of default classifier is : {}".format(defacc))
    


# In[ ]:




