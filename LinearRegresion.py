from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

def linear_regression_model():
    """
    This function reads the data from data.csv file, splits the data into training and testing sets,
    :return: the learned weights (coefficients) of the linear regression model
    """
    data = pd.read_csv('data.csv')
    
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    print(f"Learned weights (coefficients): {model.coef_}")
    return model.coef_

if __name__ == '__main__':
    linear_regression_model()