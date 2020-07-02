from flair.data import Sentence
from flair.models import SequenceTagger

# load the NER tagger
tagger = SequenceTagger.load('ner')


def entity_recognition(input_string: str):
    """
    Classifies strings into one of three classes (company name, locations, unknown_soup).
    :param input_string: string to be classified
    :return: entity id
    """

    instance = Sentence(input_string)
    tagger.predict(instance)
    to_string = instance.to_tagged_string()

    if "ORG" in to_string:
        return 1
    elif "LOC" in to_string and not has_digit(input_string):
        return 2
    else:
        return 3


def has_digit(string: str):
    """
    Checks if "Locations" entity has any digits - an attempt to minimize NER's errors
    :param string: entity name
    :return: bool
    """
    return any(char.isdigit() for char in string)


