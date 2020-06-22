
def has_digit(string: str):
    """
    Checks if "Locations" entity has any digits - an attempt to minimize NER's errors
    :param string: entity name
    :return: bool
    """
    return any(char.isdigit() for char in string)


