import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN
from jaro import jaro_winkler_metric as jaro
from leven import levenshtein
import json


class Algorithms:

    def __init__(self, print_output=False, save_output=False):
        """
        Clustering algorithm constructor
        :param print_output: prints clusters to the console
        :param save_output: save the clusters in a json file
        """
        self.print_output = print_output
        self.save_output = save_output

    def affinity_propagation(self, entity_group: set, metric: str, damping: float, preference: int, json_path=None):
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
        :param entity_group: company_names, locations, or  unknown_soup for everything else.
        :param metric: distance/similarity matrix - jaro or levenshtein.
        :param damping: damps the responsibility and availability messages
        to avoid numerical oscillations when updating these messages.
        :param preference: controls how many exemplars are used.
        :param json_path: where to save the results, default is in the folder "results" accessible from the root.
        :return: nothing for the time being.
        """

        # keep metric in string form for the json name before it's being assinged the actual ram address of the function
        str_metric = metric

        if metric == "jaro":
           metric = jaro
        elif metric == "levenshtein":
            metric = levenshtein
        else:
            print("Available metrics: [jaro, levenshtein]")
            return

        words = np.asarray(list(entity_group))

        # affprop works with similarity matrix, mult by -1 to convert from distance to similarity
        similarity_matrix = -1 * np.array([[metric(w1, w2) for w1 in words] for w2 in words])

        # precomputed because we are passing the similarity_matrix manually
        affprop = AffinityPropagation(affinity="precomputed",
                                      damping=damping,
                                      preference=preference,
                                      random_state=None)
        affprop.fit(similarity_matrix)

        clusters = {}
        for cluster_id in np.unique(affprop.labels_):

            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            clusters.update({exemplar: cluster_str})

            if self.print_output:
                print(f"- **{exemplar}** --> {cluster_str}")

        if self.save_output:
            if json_path is not None:
                with open(json_path, "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)
            else:
                with open(f"results/affinity_{str_metric}_{str(damping)}_{str(preference)}.json", "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)

    def dbscan(self, entity_group: set,  metric: str, epsilon: float, min_samples: int, json_path=None):
        """
        Density-based clustering works by identifying “dense” clusters of points, allowing it to learn clusters of
        arbitrary shape and identify outliers in the data. The general idea behind ɛ-neighborhoods is given a data
        point, we want to be able to reason about the data points in the space around it. Formally, for some
        real-valued ɛ > 0 and some point p, the ɛ-neighborhood of p is defined as the set of points that are at most
        distance ɛ away from p. In 2D space, the ɛ-neighborhood of a point p is the set of points contained in a circle
         of radius ɛ, centered at p.
        :param entity_group: name of the entity group (company name, location, unknown soup).
        :param metric: jaro or levenshtein.
        :param epsilon: ɛ, The radius (size) of the neighborhood around a data point p.
        :param min_samples: The minimum number of data points that have to be withing that neighborhood for a point
        to be considered a core point (of that given cluster ) - cluster density level threshold.
        :param json_path: where to save the results, default is in the folder "results" accessible from the root.
        :return: nothing for the time being.
        """

        # keep metric in string form for the json name before it's being assinged the actual ram address of the function
        str_metric = metric
        if metric == "jaro":
            metric = jaro
        elif metric == "levenshtein":
            metric = levenshtein
        else:
            print("Available metrics: [jaro, levenshtein]")
            return

        words = list(entity_group)

        distance_matrix = np.array([[metric(w1, w2) for w1 in words] for w2 in words])

        model = DBSCAN(eps=epsilon,
                       min_samples=min_samples,
                       algorithm='brute',
                       metric='precomputed')
        model.fit(distance_matrix)

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
            if json_path is not None:
                with open(json_path, "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)
            else:
                with open(f"results/dbscan_{str_metric}_{str(epsilon)}_{str(min_samples)}.json", "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)

    def agglomerative(self, entity_group: set,  metric: str, linkage: str, distance_threshold=None, json_path=None):
        """
        Work in progress...
        In agglomerative algorithms, each item starts in its own cluster and the two most similar items are then
        clustered. We continue accumulating the most similar items or clusters together two at a time until
        there is one cluster.
        :param entity_group:
        :param metric:
        :param linkage: {“ward”, “complete”, “average”, “single”}, default=”ward” .Which linkage criterion to use.
        The linkage criterion determines which distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
                * ward minimizes the variance of the clusters being merged.
                * average uses the average of the distances of each observation of the two sets.
                * complete or maximum linkage uses the maximum distances between all observations of the two sets.
                * single uses the minimum of the distances between all observations of the two sets.
        :param distance_threshold: float, default=None. The linkage distance threshold above which, clusters will not
        be merged. If not None, n_clusters must be None and compute_full_tree must be True.
        :param json_path: where to save the results, default is in the folder "results" accessible from the root.
        :return: nothing for the time being.
        """

        # keep metric in string form for the json name before it's being assinged the actual ram address of the function
        str_metric = metric
        if metric == "jaro":
            metric = jaro
        elif metric == "levenshtein":
            metric = levenshtein
        else:
            print("Available metrics: [jaro, levenshtein]")
            return

        data = list(entity_group)
        data = np.asarray(data)

        distance_matrix = np.array([[metric(w1, w2) for w1 in data] for w2 in data])

        agg = AgglomerativeClustering(affinity='precomputed',
                                      linkage=linkage)

        agg.fit(distance_matrix)

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
            if json_path is not None:
                with open(json_path, "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)
            else:
                with open(f"results/agglomerative_{str_metric}_{str(linkage)}.json", "w") as out:
                    json.dump(clusters, out, indent=4, sort_keys=True)
