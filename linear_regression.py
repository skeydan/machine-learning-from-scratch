from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import datasets
from sklearn import metrics

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
            
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        #print('gradient magnitude: {}\n'.format(gradient_magnitude))
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


# ------------------------------------------------------------------------------- 
#                      Test                                                     #
# -------------------------------------------------------------------------------


X, y, coef = datasets.make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, coef=True, random_state=1)
print('Actual coefficients: {}\n'.format(coef))

ones = np.ones((len(X),1))
X = np.append(ones, X, axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.2)

print('\nTraining data (X/y): \n{}\n{}\n'.format(X_train[:5], y_train[:5]))

linear_model = linear_model.LinearRegression()
linear_model.fit(X_train, y_train)

print('Coefficients (scikit-learn): {}\n'.format(linear_model.coef_))
print('Intercept (scikit-learn): {}\n'.format(linear_model.intercept_))
print('Accuracy (scikit-learn train): {}\n'.format(metrics.r2_score(y_train, linear_model.predict(X_train))))
print('Accuracy (scikit-learn test): {}\n'.format(metrics.r2_score(y_test, linear_model.predict(X_test))))

initial_weights = np.zeros(X_train.shape[1])
step_size = 1e-7
tolerance = 1e-3
weights = regression_gradient_descent(X_train, y_train, initial_weights, step_size, tolerance)

print('Coefficients (from scratch): {}\n'.format(weights))
print('Accuracy (from scratch train): {}\n'.format(metrics.r2_score(y_train, predict_output(X_train, weights))))
print('Accuracy (from scratch test): {}\n'.format(metrics.r2_score(y_test, predict_output(X_test, weights))))
