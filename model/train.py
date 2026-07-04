import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('data/loan_data.csv')
X=df.drop("Approved",axis=1)
y=df["Approved"]

X_train ,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42,max_depth=4)
model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print(f"Model Accuracy:{accuracy}")
joblib.dump(model,'model/loan_model.pkl')
print("Model saved as loan_model.pkl")