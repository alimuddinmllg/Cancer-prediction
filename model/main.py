import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    print (X)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    #train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print ('Accuracy of our model: ',accuracy_score(y_test, y_pred))
    print ('Classification report: ', classification_report(y_test,y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv('../data/data.csv')

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    print ('model',model)
    print ('scaler',scaler)
    with open ('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    # print (data)

    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Inspect the model
    print("Model type:", type(model))  # Should show sklearn.linear_model.LogisticRegression
    print("Model coefficients:", model.coef_)  # Coefficients of the logistic regression
    print("Model intercept:", model.intercept_) # Intercept of the model

    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Inspect the scaler
    print("Scaler type:", type(scaler))  # Should show sklearn.preprocessing._data.StandardScaler
    print("Scaler mean:", scaler.mean_)  # Mean used for scaling
    print("Scaler scale:", scaler.scale_)  # Standard deviation used for scaling

if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
