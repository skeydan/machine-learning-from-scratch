from __future__ import division
import numpy as np

'''
Compute logistic regression coefficients using gradient ascent
'''

# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

# compute score
def score(X, coef):
  score = np.dot(X, coef)
  return score

# link function: logistic
def logistic(score):
  prob = 1 / (1 + exp(-score))
  return prob

# squeeze score into 0-1 range using logistic link function
def predict_probability(X, coef):
    score = score(X, coef)
    predictions = logistic(score)
    return predictions

# predict class probability dependent on score (> 0 or <=0)   
def predict(X, coef):
    predicted_probs = score(X, coef)
    threshold_at_0 = np.vectorize(lambda x: 1 if x > 0 else -1)
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
def logistic_regression(X, y, coefs, l2_penalty, step_size, max_iter):
  
    for itr in xrange(max_iter):
        predictions = predict_probability(X, coefs)
        indicator = (sentiment==+1)
        errors = indicator - predictions
        
        for j in xrange(len(coefs)):           
            feature_derivative = compute_feature_derivative(errors, X[:,j], coefs[j], l2_penalty, True if j==0 else False)
            coefs[j] = coefs[j] + step_size * feature_derivative
        if itr <= 10 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            log_likelihood = compute_log_likelihood(X, y, coefs, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % (int(np.ceil(np.log10(max_iter))), itr, log_likelihood)
              
    return coefs  


  
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
