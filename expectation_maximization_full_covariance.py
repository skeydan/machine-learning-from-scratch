import sframe                                                 
import numpy as np                                            
import matplotlib.pyplot as plt                               
from scipy.stats import multivariate_normal                 
import copy   
import io
from PIL import Image
import matplotlib.mlab as mlab


'''

Clustering using expectation maximization / mixture of Gaussians

This implementation works with full covariance matrices for the cluster-specific feature distributions.

'''


# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
    return ll
        
        
def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize cluster responsibilities and compute initial loglikelihood
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for i in range(maxiter):
        print("Iteration %s" % i)
        
        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        for j in range(num_data):
            for k in range(num_clusters):
                # prior * likelihood 
                resp[j, k] = weights[k] * multivariate_normal.pdf(data[j], means[k], covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis] # evidence (likelihood of the data)
        resp = resp / row_sums 

        # Compute the total responsibility assigned to each cluster
        counts = np.sum(resp, axis=0)
        
         # M-step
        for k in range(num_clusters):
            
            # M-step update rule for cluster weight
            weights[k] = counts[k] / np.sum(counts)
            
            # M-step update rule for means
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += resp[j, k] * data[j]
            means[k] = weighted_sum / counts[k]
            
            # M-step update rule for covariances
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                difference = data[j] - means[k]
                weighted_sum += resp[j, k] * np.outer(difference.T, difference)
            covariances[k] = weighted_sum / counts[k]
            
        #print 'weights: {}\n'.format(weights)   
        #print 'means: {}\n'.format(means)
        #print 'covariances: {}\n'.format(covariances)
          
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out        
  
def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------- 
#                      Test data                                                #
# -------------------------------------------------------------------------------

# generate sample data to test on
def generate_sample_data(num_data, means, covariances, weights):
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  randomly pick a cluster id 
        k = np.random.choice(len(weights), 1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data.append(x)
    return data
  
# sample data means
init_means = [
    [5, 0], 
    [1, 1], 
    [0, 5]  
]

# sample data covariances
init_covariances = [
    [[.5, 0.], [0, .5]], 
    [[.92, .38], [.38, .91]], 
    [[.5, 0.], [0, .5]]  
]

# sample cluster weights
init_weights = [1/4., 1/2., 1/4.]  

# Generate data
np.random.seed(4)
data = generate_sample_data(100, init_means, init_covariances, init_weights)

plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------- 
#                      Run test                                                 #
# -------------------------------------------------------------------------------

np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM 
results = EM(data, initial_means, initial_covs, initial_weights)  
print results


# Plot contours after different numbers of iterations                
# after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters')

# after 12 iterations
results = EM(data, initial_means, initial_covs, initial_weights, maxiter=12) 
plot_contours(data, results['means'], results['covs'], 'Clusters after 12 iterations')

# after running EM to convergence
results = EM(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters')


# plot loglikelihoods
loglikelihoods = results['loglik']

plt.figure()
plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.plot()
plt.show()
