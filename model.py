import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Data.csv")   ## respective data 
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')   ## change accrordingly         
X = X.astype('int')    ## change accordingly type 
# y = y.astype(str)
# X = X.astype(str)
# y = y.astype(bool)
# X = X.astype(bool)

# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]  ## CUSTOMIZEX IT ACCORIDING TO THE VALUES IN YOUR CSV FILE CURRENTLY ITS 45 32 60 NUMBERS 
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


