import pickle
from sklearn.svm import SVC  # support vector classification ile dataseti sınıflandırmak için kullanıyoruz
from sklearn.metrics import classification_report  # eğitim sonuçlarını görebilmek için
import pandas as pd  # eğitim sonuçlarını pandas frame ile görüntüleyebilmek için
import os

path = os.path.dirname(__file__)
os.chdir(path)


def mnistLoad():
    with open('mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']


train_x, train_y, test_x, test_y = mnistLoad()

train_x, train_y, test_x, test_y = [pd.DataFrame(x) for x in [train_x, train_y, test_x, test_y]]

train_x = train_x / 255.0
test_x = test_x / 255.0

svc = SVC()

svc.fit(train_x, train_y.values.flatten())

filename = "modelSvm.pkl"
pickle.dump(svc, open(filename, 'wb'))

y_pred = svc.predict(test_x)
print(classification_report(test_y, y_pred))
