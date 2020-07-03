import sys
import pprint
from pathlib import Path
from varname import nameof

sys.path.insert(0, str(Path.cwd()).replace("graph_representation", ""))

from utils.helpers import entity_recognition
from utils.clustering_algorithms import Algorithms

to_be_clustered = open("../strings.txt", 'r')

# our entity groups
company_names = []
locations = []
unknown_soup = []

all_entities = []

clustering_algorithms = Algorithms(representation='graph_representation', print_output=False, save_output=True)

for string in to_be_clustered:

    string = string.replace('\n', '')
    entity = entity_recognition(string)

    if entity == 1 and string not in company_names:
        company_names.append(string)

        # Team A approach - Naive
        # logging this in order to compare approaches later
        all_entities.append(string)

        if len(company_names) < 2:
            continue

        final_company_names_clusters = clustering_algorithms.affinity_propagation(entity_group=company_names,
                                                                                  metric="jaro",
                                                                                  damping=.5,
                                                                                  preference=10,
                                                                                  entity_name=nameof(company_names)
                                                                                  )
    if entity == 2 and string not in locations:
        locations.append(string)

        # Team A approach -Naive
        # logging this in order to compare approaches later
        all_entities.append(string)

        if len(locations) < 2:
            continue

        final_locations_clusters = clustering_algorithms.affinity_propagation(entity_group=locations,
                                                                              metric="levenshtein",
                                                                              damping=.5,
                                                                              preference=10,
                                                                              entity_name=nameof(locations)
                                                                              )
    if entity == 3 and string not in unknown_soup:
        unknown_soup.append(string)

        # Team A approach -Naive
        # logging this in order to compare approaches later
        all_entities.append(string)

        if len(unknown_soup) < 2:
            continue

        final_unknown_soup_clusters = clustering_algorithms.dbscan(entity_group=unknown_soup,
                                                                   metric="jaro",
                                                                   epsilon=4,
                                                                   min_samples=1,
                                                                   entity_name=nameof(unknown_soup)
                                                                   )

    # clustering without Named Entity Recognition
    # logging this in order to compare approaches later
    all_clusters = clustering_algorithms.affinity_propagation(entity_group=all_entities,
                                                              metric="jaro",
                                                              damping=.5,
                                                              preference=3,
                                                              entity_name=nameof(all_entities)
                                                              )

# need to catch errors in case clusters are empty
print("\n--> final_company_names:")
pprint.pprint(final_company_names_clusters)

print("\n--> final_locations:")
pprint.pprint(final_locations_clusters)

print("\n--> final_unknown_soup:")
pprint.pprint(final_unknown_soup_clusters)

print("\n--> all_clusters:")
pprint.pprint(all_clusters)

to_be_clustered.close()
