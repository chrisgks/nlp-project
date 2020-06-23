from flair.data import Sentence
from flair.models import SequenceTagger
from clustering_algorithms import Algorithms
from helpers import has_digit
import pprint

# load the NER tagger
tagger = SequenceTagger.load('ner')

# our entity groups
company_names = set()
locations = set()
unknown_soup = set()


def entity_recognition(input_string: str):
    """
    Classifies strings into one of three classes (company name, locations, unknown_soup).
    :param input_string: string to be classified
    :return: nothing - just adds elements to entity sets
    """

    instance = Sentence(input_string)
    tagger.predict(instance)
    to_string = instance.to_tagged_string()

    if "ORG" in to_string:
        company_names.add(input_string)
        return 1
    elif "LOC" in to_string and not has_digit(input_string):
        locations.add(input_string)
        return 2
    else:
        unknown_soup.add(input_string)
        return 3


to_be_clustered = open("strings.txt", 'r')

selection = Algorithms(print_output=False, save_output=True, )

for idx, string in enumerate(to_be_clustered):

    flag = entity_recognition(string.replace('\n', ''))

    if flag == 1 and len(company_names) >= 2:
        final_company_names_clusters = selection.dbscan(entity_group=company_names,
                                                        metric="levenshtein",
                                                        epsilon=.3,
                                                        min_samples=1,
                                                        json_path="iterative_results",
                                                        set_name="company_names")
    elif flag == 2 and len(locations) >= 2:
        final_locations_clusters = selection.dbscan(entity_group=locations,
                                                    metric="levenshtein",
                                                    epsilon=.3,
                                                    min_samples=1,
                                                    json_path="iterative_results",
                                                    set_name="locations")
    elif flag == 3 and len(unknown_soup):
        final_unknown_soup_clusters = selection.dbscan(entity_group=unknown_soup,
                                                       metric="levenshtein",
                                                       epsilon=.3,
                                                       min_samples=1,
                                                       json_path="iterative_results",
                                                       set_name="unknown_soup")

# need to catch errors in case clusters are empty
print("\n--> final_company_names:")
pprint.pprint(final_company_names_clusters)

print("\n--> final_locations:")
pprint.pprint(final_locations_clusters)

print("\n--> final_unknown_soup:")
pprint.pprint(final_unknown_soup_clusters)

to_be_clustered.close()
