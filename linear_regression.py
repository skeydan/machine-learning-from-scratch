from __future__ import division
import numpy as np

'''
Compute linear regression coefficients using gradient descent
'''

# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

def feature_derivative(errors, feature):
    derivative = 2 * np.dot(errors, feature)
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) 
    
    while not converged:
      
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0 
        
        for i in range(len(weights)): 
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += derivative**2
            weights[i] -= step_size*derivative
            
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


  
'''
Test with smarket data and compare against scikit-learn
'''

smarket = pd.read_csv('Smarket.csv').iloc[:,1:]
smarket['dir_0_1'] = np.where(smarket['Direction'] == 'Up', 1, 0)

x_columns = ['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']
X_train = smarket[smarket['Year'] != 2005][x_columns].values
y_train = smarket[smarket['Year'] != 2005][['dir_0_1']].values[:,0]
X_test = smarket[smarket['Year'] == 2005][x_columns].values
y_test = smarket[smarket['Year'] == 2005][['dir_0_1']].values[:,0]


X_train = np.c_[X_train, np.ones(X_train.shape[0])]

logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train, y_train)
print('Coefficients ({}): {}\n'.format(x_columns, logistic_model.coef_))

initial_coefs = np.zeros(X_train.size)
step_size = 1e-7
max_iter = 500
coefs = logistic_regression(X_train, y_train, initial_coefs, max_iter)
