from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

def model_training(input_x, input_y, model, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=test_size, random_state=42)
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    train_acc = accuracy_score(y_train, y_train_predict)
    test_acc = accuracy_score(y_test, y_test_predict)

    print(f'Train acc: {train_acc}')
    cm_train = confusion_matrix(y_true=y_train, y_pred=y_train_predict)
    print(cm_train)
    print(classification_report(y_true=y_train, y_pred=y_train_predict))

    print(f'\nTest acc: {test_acc}')
    cm_test = confusion_matrix(y_true=y_test, y_pred=y_test_predict)
    print(cm_test)
    print(classification_report(y_true=y_test, y_pred=y_test_predict))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                  display_labels=model.classes_)
    disp.plot()
    return test_acc, model


def RFE_analysis(input_x, input_y, model, n_feature_to_select=0.5):
    rfe = RFE(model, step=300, n_features_to_select=n_feature_to_select)
    rfe.fit(input_x, input_y.values.ravel())
    print(input_x.columns)
    print(rfe.support_)
    print(rfe.ranking_)
    return rfe.get_support(1)


