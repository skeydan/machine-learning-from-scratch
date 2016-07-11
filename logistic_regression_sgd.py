from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

'''

Compute logistic regression coefficients using stochastic gradient ascent

This will work for binary classification tasks, and presupposes the class labels are 0 and 1.

'''


'''

# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

'''

def feature_derivative(errors, feature):  
    feature_derivative = np.dot(errors, feature)
    return feature_derivative

def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1 / (1 + np.exp(-score))
    return predictions
  
def predict(feature_matrix, coefficients):
    predicted_probs = np.dot(feature_matrix, coefficients)
    cutoff_at_0 = np.vectorize(lambda x: 1 if x > 0 else 0)
    predicted_classes = cutoff_at_0(predicted_probs)
    return predicted_classes
    
def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)
    return lp
  
def logistic_regression_SG(feature_matrix, sentiment, coefficients, step_size, batch_size, max_iter):
    log_likelihood_all = []
    
    np.random.seed(seed=1)
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]
    
    i = 0 
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix[i:i+batch_size,:], coefficients)
        indicator = sentiment[i:i+batch_size]==+1
        errors = indicator - predictions
        for j in xrange(len(coefficients)): 
            derivative = feature_derivative(errors, feature_matrix[i:i+batch_size,j])
            coefficients[j] = coefficients[j] + 1./batch_size *step_size * derivative
        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], sentiment[i:i+batch_size], coefficients)
        log_likelihood_all.append(lp)
        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
         or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print 'Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp)        
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0
    return coefficients, log_likelihood_all
  
  
def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')
    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})  

  
# ------------------------------------------------------------------------------- 
#                      Test                                                     #
# -------------------------------------------------------------------------------

  
X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=1000)

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
batch_size = 10
num_passes = 100
num_iterations = num_passes * int(len(X_train)/batch_size)

coefficients_sgd = {}
log_likelihood_sgd = {}
for step_size in np.logspace(-4, 2, num=7):
  coefficients_sgd[step_size], log_likelihood_sgd[step_size] = logistic_regression_SG(X_train, y_train, initial_coefs,
                                                                step_size=step_size, batch_size=batch_size, max_iter=num_iterations) 
  
plt.figure()    
for step_size in np.logspace(-4, 2, num=7):
    make_plot(log_likelihood_sgd[step_size], len_data=len(X_train), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)
plt.show()

plt.figure()    
for step_size in np.logspace(-4, 2, num=7)[4:]:
    make_plot(log_likelihood_sgd[step_size], len_data=len(X_train), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)
plt.show()

for step_size in np.logspace(-4, 2, num=7):
  print('Step size: {}\n'.format(step_size))
  print('Coefficients (from scratch): {}\n'.format(coefficients_sgd[step_size]))
  print('Accuracy (from scratch train): {}\n'.format(metrics.accuracy_score(y_train, predict(X_train, coefficients_sgd[step_size]))))
  print('Accuracy (from scratch test): {}\n'.format(metrics.accuracy_score(y_test, predict(X_test, coefficients_sgd[step_size]))))

    






