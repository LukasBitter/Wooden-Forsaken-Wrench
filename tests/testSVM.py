import numpy as np
import cv2

class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])


print "training with rectangle"
print ""
print "1 _______________1"
print " |               |"
print " |               | category 0"
print " |_______________|"
print " |               |"
print " |               | category 1"
print " |_______________|"
print "0               1"
print ""
samples = np.array([(0.0, 0.0),(0.0, 1.0),(1.0, 1.0),(1.0,0.0)], dtype = np.float32)
y_train = np.array([1.,0.,0.,1.], dtype = np.float32)
y_predict = np.array([(0.2, 0.3),(1.0, 0.8)], dtype = np.float32)
print "SVM, where is (0.2, 0.3) ? and  (1.0, 0.8) ?"

clf = SVM()
clf.train(samples, y_train)
y_val = clf.predict(y_predict)

for i in range(0, len(y_predict)):
    print y_predict[i], " is in category ", y_val[i]
