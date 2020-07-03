import sys
import pprint
from pathlib import Path
from varname import nameof

sys.path.insert(0, str(Path.cwd()).replace("vector_representation", ""))

from utils.clustering_algorithms import Algorithms
from utils.helpers import entity_recognition
from utils import helpers


to_be_clustered = open("../strings.txt", 'r')

# our entity groups
company_names = []
company_embeddings = []
locations = []
locations_embeddings = []
unknown_soup = []
unknown_soup_embeddings = []

all_entities = []
all_entities_embeddings = []

clustering_algorithms = Algorithms(representation='vector_representation', print_output=False, save_output=True)
selected_base_models = ['flair_forward', 'flair_backward', 'glove']

for string in to_be_clustered:

    string = string.replace('\n', '')
    entity = entity_recognition(string)

    if entity == 1 and string not in company_names:
        company_names.append(string)
        embedding = helpers.vectorize(string, selected_base_models)
        company_embeddings.append(embedding)

        # Team A approach -Naive
        # logging this in order to compare approaches later
        all_entities.append(string)
        all_entities_embeddings.append(embedding)

        if len(company_names) < 2:
            continue

        final_company_names_clusters = clustering_algorithms.dbscan(entity_group=company_names,
                                                                    epsilon=5,
                                                                    metric='euclidean',
                                                                    min_samples=1,
                                                                    embeddings=company_embeddings,
                                                                    entity_name=nameof(company_names),
                                                                    selected_base_models=selected_base_models)
    if entity == 2 and string not in locations:
        locations.append(string)
        embedding = helpers.vectorize(string, selected_base_models)
        locations_embeddings.append(embedding)

        # Team A approach -Naive
        # logging this in order to compare approaches later
        all_entities.append(string)
        all_entities_embeddings.append(embedding)

        if len(locations) < 2:
            continue

        final_locations_clusters = clustering_algorithms.dbscan(entity_group=locations,
                                                                metric='euclidean',
                                                                epsilon=6.5,
                                                                min_samples=1,
                                                                embeddings=locations_embeddings,
                                                                entity_name=nameof(locations),
                                                                selected_base_models=selected_base_models)
    if entity == 3 and string not in unknown_soup:
        unknown_soup.append(string)
        embedding = helpers.vectorize(string, selected_base_models)
        unknown_soup_embeddings.append(embedding)

        # Team A approach -Naive
        # logging this in order to compare approaches later
        all_entities.append(string)
        all_entities_embeddings.append(embedding)

        if len(unknown_soup) < 2:
            continue

        final_unknown_soup_clusters = clustering_algorithms.dbscan(entity_group=unknown_soup,
                                                                   metric='euclidean',
                                                                   epsilon=2.5,
                                                                   min_samples=1,
                                                                   embeddings=unknown_soup_embeddings,
                                                                   entity_name=nameof(unknown_soup),
                                                                   selected_base_models=selected_base_models)
    # clustering without Named Entity Recognition
    # logging this in order to compare approaches later
    all_clusters = clustering_algorithms.dbscan(entity_group=all_entities,
                                                metric='euclidean',
                                                epsilon=4.5,
                                                min_samples=1,
                                                embeddings=all_entities_embeddings,
                                                entity_name=nameof(all_entities),
                                                selected_base_models=selected_base_models)

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
