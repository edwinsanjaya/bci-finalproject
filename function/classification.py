from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def model_training(input_x, input_y, model, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size, random_state=42)
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    train_acc = accuracy_score(y_train, y_train_predict)
    test_acc = accuracy_score(y_test, y_test_predict)