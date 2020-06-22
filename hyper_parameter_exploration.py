# a separate file dedicated to experimentation and exploration of algorithm-parameter combinations
# would be nice to do this automatically, with loops etc.
# there are algorithms for finding "optimal" parameter values (i.e. elbow method for k means)
# but for now let's just try some combinations manually

from clustering_algorithms import Algorithms


to_be_clustered = open("strings.txt", 'r')

data = set()
for string in to_be_clustered:
    data.add(string.replace('\n', ''))

to_be_clustered.close()

data_set = set(data)


selection = Algorithms(print_output=True, save_output=True)


print("Agglomerative - jaro - complete")
selection.agglomerative(entity_group=data, metric="jaro", linkage="complete")
print('#'*50)

print("Agglomerative - jaro - average")
selection.agglomerative(entity_group=data, metric="jaro", linkage="average")
print('#'*50)

print("Agglomerative - levenshtein - complete")
selection.agglomerative(entity_group=data, metric="levenshtein", linkage="complete")
print('#'*50)

print("Agglomerative - levenshtein - average")
selection.agglomerative(entity_group=data, metric="levenshtein", linkage="average")
print('#'*50)


#######################################################################################################################


print("Affinity Propagation - jaro - damping=.5 - preference=5")
selection.affinity_propagation(entity_group=data, metric="jaro", damping=.5, preference=5)
print('#'*50)

print("Affinity Propagation - levenshtein - damping=.5 - preference=5")
selection.affinity_propagation(entity_group=data, metric="levenshtein", damping=.5, preference=5)
print('#'*50)

print("Affinity Propagation - jaro - damping=.8 - preference=0")
selection.affinity_propagation(entity_group=data, metric="jaro", damping=.8, preference=0)
print('#'*50)


#######################################################################################################################


print("DBSCAN - jaro - epsilon=.3 - min_samples=1")
selection.dbscan(entity_group=data, metric="jaro", epsilon=.3, min_samples=1)
print('#'*50)

print("DBSCAN - levenshtein - epsilon=.3 - min_samples=1")
selection.dbscan(entity_group=data, metric="levenshtein", epsilon=.3, min_samples=1)
print('#'*50)

print("DBSCAN - jaro - epsilon=3 - min_samples=1")
selection.dbscan(entity_group=data, metric="jaro", epsilon=.3, min_samples=1)
print('#'*50)

print("DBSCAN - levenshtein - epsilon=3 - min_samples=1")
selection.dbscan(entity_group=data, metric="levenshtein", epsilon=.3, min_samples=1)
print('#'*50)
