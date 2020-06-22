from flair.data import Sentence
from flair.models import SequenceTagger
from clustering_algorithms import Algorithms
from helpers import has_digit

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
    elif "LOC" in to_string and not has_digit(input_string):
        locations.add(input_string)
    else:
        unknown_soup.add(input_string)


to_be_clustered = open("strings.txt", 'r')

for string in to_be_clustered:
    entity_recognition(string.replace('\n', ''))

selection = Algorithms(print_output=True)

# the idea here is to apply different clustering algorithms to different types of entities
if len(company_names) >= 2:
    selection.dbscan(entity_group=company_names, metric="jaro", epsilon=.3, min_samples=1)
if len(locations) >= 2:
    selection.affinity_propagation(entity_group=locations, metric="jaro", damping=.5, preference=5)
if len(unknown_soup) >= 2:
    selection.agglomerative(entity_group=unknown_soup, metric="levenshtein", linkage="complete")

to_be_clustered.close()
