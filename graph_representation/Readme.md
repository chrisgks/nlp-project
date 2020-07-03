### Parameters, algorithms and naming conventions that support pre-computed distance matrices

Currently implemented clustering algorithms that can handle "graph representations":
* dbscan
    * parameters: **string_representation, entity_group, metric, epsilon, min samples, 
    entity name**
* affinity propagation
    * parameters: **string_representation, entity_group, metric, damping, preference,
    entity_name**
* agglomerative clustering
    * parameters: **string_representation, entity_group, metric, distance_threshold, 
    compute_full_tree, entity_name**

Available string similarity metrics:
* levenshtein distance
* jaro-winkler
* no metric combination support here



##### Examples
```g_affinity_jaro_0.5_3_all_entities_.json```
* g: **string representation prefix** - g for graph,
* affinity: clustering algorithm used, retrieved from the name of the algorithm
* jaro: **metric** used used parameter,
* 0.5: **damping** parameter,
* 3: **preference** parameter,
* all entities: **entity name** parameter


```g_affinity_levenshtein_0.5_10_locations_.json```
same as the previous one but with different parameters



```g_dbscan_jaro_3_1_unknown_soup_.json```
* g: **string representation prefix** - g for graph,
* dbscan: clustering algorithm used, retrieved from the name of the algorithm
* jaro: **metric** used used parameter,
* 3: **epsilon** parameter,
* 1: **min_samples** parameter,
* unknown_soup: **entity name** parameter


```g_dbscan_levenshtein_5_1_company_names_.json```
same as the previous one but with different parameters

**NOTE**: I haven't found any combination using string similarity as a metric that gives good results _yet_.
