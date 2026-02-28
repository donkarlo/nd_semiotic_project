from typing import Protocol

from nd_robotic.robot.robot import Meaning


class FormationSubscriber(Protocol):
    def do_with_formed_meaning(self, meaning:Meaning) -> Meaning: ...