from abc import ABC, abstractmethod

from nd_robotic.robot.robot import \
    Corpus


class Structed(Corpus, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_plain_text(self)->str:
        pass
