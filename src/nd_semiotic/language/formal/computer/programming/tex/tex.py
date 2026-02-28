from nd_semiotic.language.formal.computer.programming.tex.composing.kind.from_tex_directory.from_tex_directory import \
    FromTexDirectory
from nd_utility.os.file_system.path.directory import Directory


class Tex:
    def compose_from_tex_source_directory(self, source_directory_path:Directory):
        composing = FromTexDirectory(source_directory_path)
        self._generate_preamble().stringify()
        self._generate_document.stringify()

