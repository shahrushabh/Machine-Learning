from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import numpy as np
X = np.array([[0.6585,0.2444], [2.246, 0.5281], [-2.7665, -3.8303], [-1.2565, 3.4912], [-0.7973, 1.2288], [1.117, 2.2637]])
Y = np.array([0, 0, 0, 1, 1, 1])
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, Y)
print "The prediction for (0,1) with QDA is: " + str(qda.predict([0,1]))

gnb = GaussianNB()
gnb.fit(X,Y)
print "The prediction for (0,1) with Naive Bayes is: " + str(gnb.predict([0,1]))
