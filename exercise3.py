import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as pyplot
from scipy.stats import multivariate_normal

##################### LOADING DATA #####################
#Load from matlab
annots = loadmat("dataset1_G_noisy.mat")

#Training sets
trn_x = annots["trn_x"]
trn_x_class = annots["trn_x_class"]
trn_y = annots["trn_y"]
trn_y_class = annots["trn_y_class"]

#Testing sets
tst_xy = annots["tst_xy"]
tst_xy_class = annots["tst_xy_class"]
tst_xy_126 = annots["tst_xy_126"]
tst_xy_126_class = annots["tst_xy_126_class"]

#Normality tests and it is normal
pyplot.hist(trn_x[:,0])
pyplot.hist(trn_y[:,0])
pyplot.show()

##################### EXERCISE A #####################
#Both are normal so estimate priors
n = trn_x_class.size + trn_y_class.size
priorx = trn_x_class.size / n
priory = trn_y_class.size / n

#Calc means
mx = trn_x.mean(0)
my = trn_y.mean(0)

#Estimate covariance matrix and transpose to keep rank at 2
cx = np.cov(trn_x.T)
cy = np.cov(trn_y.T)

#Calc multivariate norm
mnx = multivariate_normal(mx, cx)
mny = multivariate_normal(my, cy)

#Calculate posteriori
postx = priorx * mnx.pdf(tst_xy)
posty = priory * mny.pdf(tst_xy)

#Classify predictions
p = np.where(postx - posty >= 0, 1, 2) # if its lower than 0 value is set 1 and 2 otherwise
s = tst_xy_class.squeeze() #throws error if value is larger than 1 so keeps only below 1
a = np.sum(p == s) / s.size
print("Acuracy of a: {:.8f}".format(a))

##################### EXERCISE B #####################
#Priors not required since its uniform

#Calc posteriori
postx = mnx.pdf(tst_xy_126)
posty = mny.pdf(tst_xy_126)

#Classify predictions
p = np.where(postx - posty >= 0, 1, 2)
s = tst_xy_126_class.squeeze()
a = np.sum(p == s) / s.size
print("Acuracy of b: {:.8f}".format(a))

##################### EXERCISE C #####################
##Setting new priors
priorx = 0.9
priory = 0.1

#Calc posteriori
postx = priorx * mnx.pdf(tst_xy_126)
posty = priory * mny.pdf(tst_xy_126)

#Classify predictions
p = np.where(postx - posty >= 0, 1, 2)
s = tst_xy_126_class.squeeze()
a = np.sum(p == s) / s.size
print("Acuracy of c: {:.8f}".format(a))
