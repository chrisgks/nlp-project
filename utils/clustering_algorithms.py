import numpy as np
import json
from jaro import jaro_winkler_metric as jaro
from leven import levenshtein
from pathlib import Path
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN

class Algorithms:

    def __init__(self, representation: str = None, print_output: bool = False, save_output: bool = False):
        """
        Clustering algorithm constructor
        :param print_output, prints clusters to the console
        :param save_output, save the clusters in a json file
        :param representation, choose string representation 'vector representation', or 'graph representation'
        """

        if representation is None or representation not in ['vector_representation', 'graph_representation']:
            raise SystemExit("[Error]: Please provide one of the available representations: "
                             "['vector_representation', 'graph_representation']")

        self.representation = representation
        self.print_output = print_output
        self.save_output = save_output

    def affinity_propagation(self, entity_group: list,
                             metric: str = None,
                             damping: float = None,
                             embeddings: list = None,
                             entity_name: str = None,
                             selected_base_models: list = None):
        """
        In contrast to other traditional clustering methods, affprop does not require you to specify the number of
        clusters. In creators' terms, in affprop, each data point sends messages to all other points informing its
        targets of each target’s relative attractiveness to the sender. Each target then responds to all senders with a
        reply informing each sender of its availability to associate with the sender, given the attractiveness of the
        messages that it has received from all other senders. Senders reply to the targets with messages informing each
        target of the target’s revised relative attractiveness to the sender, given the availability messages it has
        received from all targets. The message-passing procedure proceeds until a consensus is reached. Once the sender
        is associated with one of its targets, that target becomes the point’s exemplar. All points with the same
        exemplar are placed in the same cluster.

        :param entity_group, company_names, locations, or  unknown_soup for everything else.
        :param metric, distance/similarity matrix - jaro or levenshtein. Will only needed when "graph representation"
         is being selected at constructor time
        :param damping, damps the responsibility and availability messagesto avoid numerical oscillations when updating
         these messages.
        :param entity_name, useful for results file naming
        :param embeddings, list or embeddings in case of vector representation
        :param  selected_base_models, need this for jason naming
        :return: clusters
        """
        words = np.asarray(entity_group)
        # graph representation aka string similarity - requires precombuted matrix of destances between strings
        # and "custom"  distance metric metric
        if self.representation == "graph_representation":
            affinity = "precomputed"
            if metric == "jaro":
                distance_matrix = np.array([[jaro(w1, w2) for w1 in words] for w2 in words])
                features = distance_matrix
            elif metric == "levenshtein":
                similarity_matrix = -1 * np.array([[levenshtein(w1, w2) for w1 in words] for w2 in words])
                features = similarity_matrix
            else:
                raise SystemExit(f"[ERROR]: function affinity_propagation() -> Provide one of the available metrics: "
                                 f"['jaro', 'levenshtein']")
        else:
            # in this case we are dealing with embeddings; python currently supports only euclidean distance
            affinity = 'euclidean'
            metric = affinity
            features = embeddings
            if selected_base_models:
                selected_base_models = selected_base_models

        affprop = AffinityPropagation(affinity=affinity, damping=damping, random_state=None)
        affprop.fit(features)

        clusters = {}
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            clusters.update({exemplar: cluster_str})

            if self.print_output:
                print(f"- **{exemplar}** --> {cluster_str}")

        if self.save_output:
            # name of json is dynamic - all parameter values are integrated in the name of the file itself
            with open(f"{str(Path.cwd())}/results/{self.representation[0]}_affinity_"
                      f"{metric}_{str(damping)}_{entity_name}_{selected_base_models}.json", "w+") as out:
                json.dump(clusters, out, indent=4, sort_keys=True)

        return clusters

    def dbscan(self, entity_group: list = None,
               metric: str = None,
               epsilon: float = None,
               min_samples: int = None,
               embeddings: list = None,
               entity_name: str = None,
               selected_base_models: list = None):
        """
        Density-based clustering works by identifying “dense” clusters of points, allowing it to learn clusters of
        arbitrary shape and identify outliers in the data. The general idea behind ɛ-neighborhoods is given a data
        point, we want to be able to reason about the data points in the space around it. Formally, for some
        real-valued ɛ > 0 and some point p, the ɛ-neighborhood of p is defined as the set of points that are at most
        distance ɛ away from p. In 2D space, the ɛ-neighborhood of a point p is the set of points contained in a circle
         of radius ɛ, centered at p.

        :param entity_group, name of the entity group (company name, location, unknown soup).
        :param metric, distance/similarity matrix - jaro or levenshtein. Will only needed when "graph representation"
         is being selected at constructor time
        :param epsilon, ɛ, the radius (size) of the neighborhood around a data point p.
        :param min_samples, the minimum number of data points that have to be withing that neighborhood for a point
        to be considered a core point (of that given cluster ) - cluster density level threshold.
        :param embeddings, list or embeddings in case of vector representation
        :param entity_name, the name of the set - helps with json naming (optional)
        :param selected_base_models, need this for jason naming
        :return: clusters
        """
        words = list(entity_group)

        # graph representation aka string similarity - requires precombuted matrix of destances between strings
        # and "custom"  distance metric metric
        if self.representation == "graph_representation":
            affinity = "precomputed"
            if metric == "jaro":
                distance_matrix = np.array([[jaro(w1, w2) for w1 in words] for w2 in words])
                features = distance_matrix
            elif metric == "levenshtein":
                similarity_matrix = -1 * np.array([[levenshtein(w1, w2) for w1 in words] for w2 in words])
                features = similarity_matrix
            else:
                raise SystemExit(f"[ERROR]: function affinity_propagation() -> Provide one of the available metrics: "
                                 f"['jaro', 'levenshtein']")
        else:
            # in this case we are dealing with embeddings; python currently supports only euclidean distance
            affinity = metric
            metric = affinity
            features = embeddings
            if selected_base_models:
                selected_base_models = selected_base_models

        model = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric)

        features = np.array(features)
        model.fit(features)

        clusters = {}
        for idx, label in enumerate(model.labels_):
            if label not in clusters.keys():
                # got an error here, needs to be int, not int64 - hence the type cast to int
                clusters.update({int(label): [words[idx]]})
            else:
                clusters[int(label)].append(words[idx])

        if self.print_output:
            for key, item in clusters.items():
                print(key, item)

        if self.save_output:
            # name of json is dynamic - all parameter values are integrated in the name of the file itself
            with open(f"{str(Path.cwd())}/results/{self.representation[0]}_dbscan_"
                      f"{metric}_{str(epsilon)}_{str(min_samples)}_{entity_name}_{selected_base_models}.json", "w+") \
                    as out:
                json.dump(clusters, out, indent=4, sort_keys=True)

        return clusters

    def agglomerative(self, entity_group: list = None,
                      metric: str = None,
                      linkage: str = None,
                      distance_threshold: float = None,
                      compute_full_tree: bool = True,
                      n_clusters: int = None,
                      embeddings: list = None,
                      entity_name: str = None,
                      selected_base_models: list = None
                      ):
        """
        In agglomerative algorithms, each item starts in its own cluster and the two most similar items are then
        clustered. We continue accumulating the most similar items or clusters together two at a time until
        there is one cluster.

        :param entity_group, name of the entity group (company name, location, unknown soup).
        :param metric, distance/similarity matrix - jaro or levenshtein. Will only needed when "graph representation"
         is being selected at constructor time
        :param linkage, {“ward”, “complete”, “average”, “single”}, default=”ward” .Which linkage criterion to use.
        The linkage criterion determines which distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
                * ward minimizes the variance of the clusters being merged.
                * average uses the average of the distances of each observation of the two sets.
                * complete or maximum linkage uses the maximum distances between all observations of the two sets.
                * single uses the minimum of the distances between all observations of the two sets.
        :param distance_threshold, float, default=None. The linkage distance threshold above which, clusters will not
        be merged. If not None, n_clusters must be None and compute_full_tree must be True.
        :param n_clusters, the number of clusters to find. It must be None if distance_threshold is not None.
        :param compute_full_tree, ‘auto’ or bool, default=’auto'. It must be True if distance_threshold is not None.
        By default compute_full_tree is “auto”, which is equivalent to True when distance_threshold is not None or that
         n_clusters is inferior to the maximum between 100 or 0.02 * n_samples. Otherwise, “auto” is equivalent to False
        :param embeddings
        :param entity_name, the name of the set - helps with json naming (optional)
        :param selected_base_models need this for jason naming
        :return: clusters
        """
        data = list(entity_group)
        data = np.asarray(data)

        # graph representation aka string similarity - requires precombuted matrix of destances between strings
        # and "custom"  distance metric metric
        if self.representation == "graph_representation":
            affinity = "precomputed"
            if metric == "jaro":
                distance_matrix = np.array([[jaro(w1, w2) for w1 in data] for w2 in data])
                features = distance_matrix
            elif metric == "levenshtein":
                similarity_matrix = -1 * np.array([[levenshtein(w1, w2) for w1 in data] for w2 in data])
                features = similarity_matrix
            else:
                raise SystemExit(f"[ERROR]: function affinity_propagation() -> Provide one of the available metrics: "
                                 f"['jaro', 'levenshtein']")
        else:
            # in this case we are dealing with embeddings; python currently supports only euclidean distance
            affinity = metric
            metric = affinity
            features = embeddings
            if selected_base_models:
                selected_base_models = selected_base_models

        agg = AgglomerativeClustering(affinity=affinity,
                                      linkage=linkage,
                                      distance_threshold=distance_threshold,
                                      compute_full_tree=compute_full_tree,
                                      n_clusters=n_clusters)
        agg.fit(features)

        clusters = {}
        for idx, label in enumerate(agg.labels_):
            if label not in clusters.keys():
                clusters.update({int(label): [data[idx]]})
            else:
                clusters[int(label)].append(data[idx])

        if self.print_output:
            for key, item in clusters.items():
                print(key, item)
        if self.save_output:
            # name of json is dynamic - all parameter values are integrated in the name of the file itself
            with open(f"{str(Path.cwd())}/results/{self.representation[0]}_agglomarative_"
                      f"{str(metric)}_{linkage}_{distance_threshold}_{compute_full_tree}_"
                      f"{entity_name}_{selected_base_models}.json", "w+") as out:
                json.dump(clusters, out, indent=4, sort_keys=True)

        return clusters
