from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

'''

Compute logistic regression coefficients using gradient ascent

This will work for binary classification tasks, and presupposes the class labels are 0 and 1.

'''


# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

# compute score
def compute_score(X, coef):
  score = np.dot(X, coef)
  return score

# link function: logistic
def logistic(score):
  prob = 1 / (1 + np.exp(-score))
  return prob

# squeeze score into 0-1 range using logistic link function
def predict_probability(X, coef):
    score = compute_score(X, coef)
    predictions = logistic(score)
    return predictions

# predict class probability dependent on score (> 0 or <=0)   
def predict(X, coef):
    predicted_probs = compute_score(X, coef)
    threshold_at_0 = np.vectorize(lambda x: 1 if x > 0 else 0)
    predicted_classes = threshold_at_0(predicted_probs)
    return predicted_classes
    
# derivative for one feature
def compute_feature_derivative(errors, feature, coefficient, l2_penalty, is_constant):  
    if is_constant:
        feature_derivative = np.dot(errors, feature)
    else:
        feature_derivative = np.dot(errors, feature) - 2 * l2_penalty * coefficient
    return feature_derivative

# log likelihood function
def compute_log_likelihood(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    log_likelihood = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    return log_likelihood

# compute logistic regression coefficients using gradient ascent
def logistic_regression(X, y, coefs, step_size, max_iter, l2_penalty=0):
  
    for itr in xrange(max_iter):
        predictions = predict_probability(X, coefs)
        indicator = y == +1
        errors = indicator - predictions
        
        for j in xrange(len(coefs)):           
            feature_derivative = compute_feature_derivative(errors, X[:,j], coefs[j], l2_penalty, True if j==0 else False)
            coefs[j] = coefs[j] + step_size * feature_derivative
        if itr <= 10 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            log_likelihood = compute_log_likelihood(X, y, coefs, l2_penalty)
            print 'iteration %*d: coefficients = %s' % (int(np.ceil(np.log10(max_iter))), itr, str(coefs))  
            print 'iteration %*d: log likelihood of observed labels = %.8f\n' % (int(np.ceil(np.log10(max_iter))), itr, log_likelihood)
                       
    return coefs  


  
# ------------------------------------------------------------------------------- 
#                      Test                                                     #
# -------------------------------------------------------------------------------


X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=100)


'''
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

X = X[:100,:2]
y = y[:100]
'''

ones = np.ones((len(X),1))
X = np.append(ones, X, axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.2)

print('\nTraining data (X/y): \n{}\n{}\n'.format(X_train[:5], y_train[:5]))

logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train, y_train)
print('Coefficients (scikit-learn): {}\n'.format(logistic_model.coef_))
print('Intercept (scikit-learn): {}\n'.format(logistic_model.intercept_))
print('Accuracy (scikit-learn train): {}\n'.format(metrics.accuracy_score(y_train, logistic_model.predict(X_train))))
print('Accuracy (scikit-learn test): {}\n'.format(metrics.accuracy_score(y_test, logistic_model.predict(X_test))))

initial_coefs = np.zeros(X_train.shape[1])
step_size = 0.5
max_iter = 100
coefs = logistic_regression(X_train, y_train, initial_coefs, step_size, max_iter)
print('Coefficients (from scratch): {}\n'.format(coefs))
print('Accuracy (from scratch train): {}\n'.format(metrics.accuracy_score(y_train, predict(X_train, coefs))))
print('Accuracy (from scratch test): {}\n'.format(metrics.accuracy_score(y_test, predict(X_test, coefs))))
