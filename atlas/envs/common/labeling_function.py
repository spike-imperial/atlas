from abc import abstractmethod, ABC
from typing import List

import chex


class LabelingFunction(ABC):
    """
    Abstraction of the labeling function, which maps low-level environment
    observations into propositions.
    """

    @abstractmethod
    def get_alphabet_size(self) -> int:
        """
        Returns the number of propositions an observation can be mapped to.
        """
        raise NotImplementedError

    def get_str_alphabet(self) -> List[str]:
        """
        Returns a list with the string representation of each proposition
        in the alphabet.
        """
        return [self.prop_to_str(i) for i in range(self.get_alphabet_size())]

    @abstractmethod
    def get_label(self, observation: chex.Array) -> chex.Array:
        """
        Returns a label, i.e. an assigment of propositions to false (-1) or true (+1)
        depending on whether they are present in the observation.
        """
        raise NotImplementedError

    def label_to_prop_list(self, label: chex.Array) -> List[str]:
        """
        Returns a list containing the string representation of each positively
        appearing proposition in the label.
        """
        arr = []
        for prop_id in range(len(label)):
            if label[prop_id] == 1:
                arr.append(self.prop_to_str(prop_id))
        return arr

    @abstractmethod
    def prop_to_str(self, prop_id: int) -> str:
        """
        Returns the string representation of a given proposition.
        """
        raise NotImplementedError
