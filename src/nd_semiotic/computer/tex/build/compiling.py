from utilix.os.file_system.file.file import File as OsFile


class Compiling:
    def __init__(self, main_file:OsFile, compiler):
        self._main_file = main_file