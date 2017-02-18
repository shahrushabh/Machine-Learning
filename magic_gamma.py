import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score
style.use('ggplot')

""" Defines all arrays to use in this Analysis. """
fLength=[]
fWidth=[]
fSize=[]
fConc=[]
fConc1=[]
fAsym=[]
fM2Long=[]
fM3Trans=[]
fAplha=[]
fDist=[]
class_label=[]
train_data=[]
train_classes=[]
clf_predicted_label=[]
poly_predicted_label=[]
lin_predicted_label=[]
true_label=[]


dataset = open('gamma.csv','r')
for line in dataset:
    data = line.split(",")
    fLength.append(data[0])
    fWidth.append(data[1])
    fSize.append(data[2])
    fConc.append(data[3])
    fConc1.append(data[4])
    fAsym.append(data[5])
    fM2Long.append(data[6])
    fM3Trans.append(data[7])
    fAplha.append(data[8])
    fDist.append(data[9])
    class_label.append(data[10].rstrip("\n"))
    
for i in range(8000):
    train_data.append([fLength[i], fWidth[i], fSize[i], fConc[i], fConc1[i], fAsym[i], fM2Long[i], fM3Trans[i], fAplha[i], fDist[i]])
    train_classes.append(class_label[i])

for i in range(14020, 19020):
    train_data.append([fLength[i], fWidth[i], fSize[i], fConc[i], fConc1[i], fAsym[i], fM2Long[i], fM3Trans[i], fAplha[i], fDist[i]])
    train_classes.append(class_label[i])

X = np.array(train_data)
Y = train_classes

clf = svm.SVC(kernel="linear", C=1.0).fit(X,Y)
poly = svm.SVC(kernel="poly", degree=3, C=1.0).fit(X,Y)
lin = svm.LinearSVC(C=1.0).fit(X,Y)


print "Generating predictions on test data:"

for i in range(8001,14019):
    predicted_label.append(clf.predict([fLength[i], fWidth[i], fSize[i], fConc[i], fConc1[i], fAsym[i], fM2Long[i], fM3Trans[i], fAplha[i], fDist[i]]))
    poly_predicted_label.append(poly.predict([fLength[i], fWidth[i], fSize[i], fConc[i], fConc1[i], fAsym[i], fM2Long[i], fM3Trans[i], fAplha[i], fDist[i]]))
    lin_predicted_label.append(lin.predict([fLength[i], fWidth[i], fSize[i], fConc[i], fConc1[i], fAsym[i], fM2Long[i], fM3Trans[i], fAplha[i], fDist[i]]))
    true_label.append(class_label[i])

print true_label
print predicted_label
print "Accuracy score for linear is computed as: " + str(accuracy_score(true_label, predicted_label));
print "Accuracy score for poly is computed as: " + str(accuracy_score(true_label, poly_predicted_label));
print "Accuracy score for LinearSVC is computed as: " + str(accuracy_score(true_label, lin_predicted_label));

w = clf.coef_[0]
a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, "k-", label="Non weighted div")
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.legend()
plt.show()
