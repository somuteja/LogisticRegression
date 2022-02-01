from sklearn.datasets import load_digits
digits = load_digits()
#8*8 images of integer pixels in the range 0 to 16
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(digits['data'],digits['target'],
                                                  test_size=0.25, random_state = 42)

from sklearn.linear_model import LogisticRegression

logisticRegression  = LogisticRegression()

logisticRegression.fit(X_train,y_train)

print("Training score {:.2f}".format(logisticRegression.score(X_train,y_train))) #training accuracy

print("Test score {:.2f}".format(logisticRegression.score(X_test,y_test))) #test accuracy

#we get test score and a training score of 0.99 and 0.97, which implies this is a good fit
