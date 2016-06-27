from __future__ import division  
import pandas as pd

train_data = pd.read_csv('data/lending-club-data-train.csv')
test_data = pd.read_csv('data/lending-club-data-test.csv')

    
'''
Tree node representation:

{ 
   'is_leaf'            : True/False.
   'prediction'         : Prediction at the leaf node.
   'left'               : (dictionary corresponding to the left tree).
   'right'              : (dictionary corresponding to the right tree).
   'splitting_feature'  : The feature that this node splits on.
}
'''

# ------------------------------------------------------------------------------- 
#                      Functions                                                #
# -------------------------------------------------------------------------------

# number of mistakes made in intermediate nodes
def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    num_pos = labels_in_node[labels_in_node==1].size
    num_neg = labels_in_node[labels_in_node==-1].size
    majority_vote = 1 if num_pos > num_neg else -1
    return num_pos if majority_vote == -1 else num_neg

# find best feature to split on  
def best_splitting_feature(data, features, target):
    
    best_feature = None 
    best_error = 1    
    num_data_points = float(len(data))  
    
    for feature in features:
        
        print("Checking feature: {}".format(feature))
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        
        left_mistakes = intermediate_node_num_mistakes(left_split[target]) 
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        error = (left_mistakes + right_mistakes) / len(data)
        print("Error: {}\n".format(error))
        if error < best_error:
          best_error = error
          best_feature = feature
          
    print("Best feature for this split: {}\n\n".format(best_feature))
    return best_feature # Return the best feature we found

# create a leaf node
def create_leaf(target_values):
  
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True   }  
    
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1         
    else:
        leaf['prediction'] = -1  
    return leaf 

# main function to create decision tree
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("\n--------------------------------------------------------------------")
    print("Subtree, depth = {} ({} data points).\n").format(current_depth, len(target_values))
    
    # Stopping conditions 
    if  intermediate_node_num_mistakes(data[target]) == 0:  
        print("No predictions at current node. Stopping recursion.") 
        return create_leaf(target_values)
    if remaining_features == []: 
        print("No more features to split on. Stopping recursion.")  
        return create_leaf(target_values)    
    
    if current_depth >= max_depth:  
        print("Reached maximum depth. Stopping recursion.")
        return create_leaf(target_values)

    splitting_feature = best_splitting_feature(data, features, target)
    left_split = data[data[splitting_feature] == 0]
    right_split =  data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print("Split on feature {}. ({}, {})\n\n").format(splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        return create_leaf(right_split[target])
      
    # Recurse
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)   
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def classify(tree, x, annotate = False):   
    if tree['is_leaf']:
        if annotate: 
            print("At leaf, predicting {}\n".format(tree['prediction']))
        return tree['prediction'] 
    else:
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print("Split on {} = {}\n".format(tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    return (prediction != data[target]).sum() / prediction.size

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    #split_feature, split_value = split_name.rsplit('_',1)
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

      
# ------------------------------------------------------------------------------- 
#                      Test                                                     #
# -------------------------------------------------------------------------------

# Binarize categorical variables
categorical_variables = []
for feat_name, feat_type in zip(train_data.columns, train_data.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)
#print categorical_variables
train_data = train_data.join(pd.get_dummies(train_data[categorical_variables]))
train_data = train_data.drop(categorical_variables,1)
test_data = test_data.join(pd.get_dummies(test_data[categorical_variables]))
test_data = test_data.drop(categorical_variables,1)

features = train_data.columns.tolist()
features.remove('safe_loans') 

decision_tree_model = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6)          
classify(decision_tree_model, test_data.iloc[0,:], annotate=True)
evaluate_classification_error(decision_tree_model, test_data, 'safe_loans')      
print_stump(decision_tree_model)
print_stump(decision_tree_model['left'], decision_tree_model['splitting_feature'])
print_stump(decision_tree_model['left']['left'], decision_tree_model['left']['splitting_feature'])

      
      
          
               
