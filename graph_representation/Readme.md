#### Naming conventions for experiments using pre-computed string similarity matrices (aka "graph representation")

Currently implemented clustering algorithms that can handle "graph representations":
* dbscan
    * parameters: string_representation, entity_group, metric, epsilon, min samples, 
    entity name
* affinity propagation
    * parameters: string_representation, entity_group, metric, damping, preference,
    entity_name
* agglomerative clustering
    * parameters: string_representation, entity_group, metric, distance_threshold, 
    compute_full_tree, entity_name

Available string similarity metrics:
* levenshtein distance
* jaro-winkler
* no metric combination support here



##### Example