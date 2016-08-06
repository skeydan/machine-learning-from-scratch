import sframe                                                  
import matplotlib.pyplot as plt                                
import numpy as np                                             
from scipy.sparse import csr_matrix                            
from sklearn.preprocessing import normalize                   
from sklearn.metrics import pairwise_distances                
import sys      
import os
import time

'''

k means clustering

'''


# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix( (data, indices, indptr), shape)
  

# Choose k initial centroids randomly from the dataset
def get_initial_centroids(data, k, seed=None):
    if seed is not None: 
        np.random.seed(seed)
    n = data.shape[0] 
    rand_indices = np.random.randint(0, n, k)
    centroids = data[rand_indices,:].toarray()
    return centroids  

# kmeans++ 
# choose initial cluster centers to be maximally apart
def smart_initialize(data, k, seed=None):
    if seed is not None: 
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()
    
    for i in xrange(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))
        centroids[i] = data[idx,:].toarray()
        distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean'),axis=1)
    return centroids

def assign_clusters(data, centroids):
    distances_from_centroids = pairwise_distances(data, centroids)
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    return cluster_assignment
  
def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in xrange(k):
        member_data_points = data[cluster_assignment == i]
        centroid = member_data_points.mean(axis=0)
        centroid = centroid.A1 # convert sparse matrix to ndarray
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids

# sum of distances to cluster center over all data points  
def compute_heterogeneity(data, k, centroids, cluster_assignment):    
    heterogeneity = 0.0
    for i in xrange(k):
        member_data_points = data[cluster_assignment==i, :]
        if member_data_points.shape[0] > 0: 
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)        
    return heterogeneity

# kmeans single run
def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in xrange(maxiter):        
        if verbose:
            print('\nIteration: {}\n'.format(itr))
        
        # 1. Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)            
        # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)            
        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        # Print number of new assignments 
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment

# run kmeans num_runs times and return best
# uses kmeans++
def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    heterogeneity = {}
    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None
    
    for i in xrange(num_runs):
        if seed_list is not None: 
            seed = seed_list[i]
            np.random.seed(seed)
        else: 
            seed = int(time.time())
            np.random.seed(seed)
        
        initial_centroids = smart_initialize(data, k, seed)
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter,
                                           record_heterogeneity=None, verbose=False)
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
    return final_centroids, final_cluster_assignment


# plot heterogeneity vs number of iterations
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.show()

# plot heterogeneity for different numbers of k
def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.show()


def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word, display_content=True):
    print('==========================================================')

    for c in xrange(k):
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        for i in xrange(5): 
            print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroids[c,idx[i]])),
        print('')
        
        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in xrange(8):
                text = ' '.join(wiki[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki[nearest_neighbors[i]]['name'],
                    distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print('==========================================================')
        


   
  
# ------------------------------------------------------------------------------- 
#                      Test                                                     #
# -------------------------------------------------------------------------------


wiki = sframe.SFrame('data/people_wiki.gl/')
tf_idf = load_sparse_csr('data/people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('data/people_wiki_map_index_to_word.gl/')

tf_idf = normalize(tf_idf)

k = 10
heterogeneity_smart = {}
start = time.time()
#for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
for seed in [0]:
    initial_centroids = smart_initialize(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()
end = time.time()



# ------------------------------------------------------------------------------- 
#                      Test with different k                                    #
# -------------------------------------------------------------------------------

start = time.time()
centroids = {}
cluster_assignment = {}
heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]
seed_list = [0]
#seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

for k in k_list:
    heterogeneity = []
    centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
                                                               num_runs=len(seed_list),
                                                               seed_list=seed_list,
                                                               verbose=True)
    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
    heterogeneity_values.append(score)

plot_k_vs_heterogeneity(k_list, heterogeneity_values)

end = time.time()
print(end-start)


# ------------------------------------------------------------------------------- 
#                      Visualize with different k                               #
# -------------------------------------------------------------------------------


k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word)
