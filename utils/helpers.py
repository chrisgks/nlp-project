from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import DocumentPoolEmbeddings
from flair.data import Sentence

# constructing all the models onnce and not loading a new one at every function call; saves a lot of ram; faster
bert_embedding = TransformerWordEmbeddings('bert-base-cased')
roberta_embedding = TransformerWordEmbeddings('roberta-base')
glove_embedding = WordEmbeddings('glove')
character_embeddings = CharacterEmbeddings()
flair_forward = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')


def vectorize(string: str, args: list = None):
    """

    :param string:
    :param args:
    :return:
    """

    if not args:
        raise SystemExit(f"[ERROR]: function {vectorize.__name__}() -> Provide at least one base model: ['bert',"
                         f"'roberta', 'glove', 'character', 'flair_forward', 'flair_backward']")

    embeddings = []

    if 'bert' in args:
        embeddings.append(bert_embedding)
    if 'roberta' in args:
        embeddings.append(roberta_embedding)
    if 'glove' in args:
        embeddings.append(glove_embedding)
    if 'character' in args:
        embeddings.append(character_embeddings)
    if 'flair_forward' in args:
        embeddings.append(flair_forward)
    if 'flair_backward' in args:
        embeddings.append(flair_backward)

    # if none of the above, then the model combination passed is not supported
    if not embeddings:
        raise SystemExit(f"[ERROR]: function {vectorize.__name__}() -> {args} not available. Supported models: "
                         f"['bert', 'roberta', 'glove', 'character', 'flair_forward', 'flair_backward']")

    stacked_embeddings = DocumentPoolEmbeddings(embeddings=embeddings,
                                                fine_tune_mode='none',
                                                pooling='mean')

    sentence = Sentence(string)
    stacked_embeddings.embed(sentence)

    return sentence.embedding.detach().numpy()


# loading the NER tagger once and not loading a new one at every function call; saves some ram; faster
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


def save_output_to_json():
    pass