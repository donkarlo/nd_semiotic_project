from nd_semiotic.language.formal.computer.programming.tex.composing.composing import Composing
from nd_utility.data.kind.dic.dic import Dic
from nd_utility.os.file_system.path.directory import Directory
from typing import Optional

class FromTexDirectory(Composing):
    def __init__(self, project_directory: Directory):
        """
        instructions:
            strat from the given directory and search for the .composing file naming exactly the same as the given directory name.

        Args:
            project_directory:
        """
        self._preamble_dic = Dic({})
        self._document_dic = Dic({})
        self._bibliography_dic = Dic({})

        self._composed_string:Optional[str] = None

    def get_composed_string(self):
        pass


if __name__ == "__main__":
    directory_path = "/home/donkarlo/Dropbox/repo/phd_journal_paper/src/kind/journal"


    composing_from_text_directory = FromTexDirectory()
    composing_from_text_directory.get_composed_string(Directory(file_path))