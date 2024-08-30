"""
This module provides sample functions and classes with detailed docstrings
to improve code documentation and readability.
"""

class SampleClass:
    """
    A sample class to demonstrate the use of docstrings.

    Attributes:
        attribute1 (str): Description of attribute1.
        attribute2 (int): Description of attribute2.
    """

    def __init__(self, attribute1, attribute2):
        """
        Initializes the SampleClass with the given attributes.

        Args:
            attribute1 (str): The first attribute.
            attribute2 (int): The second attribute.
        """
        self.attribute1 = attribute1
        self.attribute2 = attribute2

    def sample_method(self, param1, param2):
        """
        A sample method that performs an operation on the attributes.

        Args:
            param1 (str): The first parameter.
            param2 (int): The second parameter.

        Returns:
            str: A formatted string combining the attributes and parameters.
        """
        return f"{self.attribute1} {param1} and {self.attribute2} {param2}"


def sample_function(param1, param2):
    """
    A sample function to demonstrate the use of docstrings.

    Args:
        param1 (str): The first parameter.
        param2 (int): The second parameter.

    Returns:
        str: A formatted string combining the parameters.
    """
    return f"Parameters are {param1} and {param2}"

