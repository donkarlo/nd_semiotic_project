# Goal

The goal of this project is to support everything related to semiotics, with natural language as its most important
branch. It provides a façade (a shared interface) for using different language models, embeddings, and vector databases.

It prioritizes the hierarchy of corpora in the file system.

For files such as YAML, JSON, etc., it considers and prioritizes the data entry hierarchies within these files.

## Language Conversion

It develops conversion tools between different languages and formats, such as translating from YAML to LaTeX files.

# Compiling

It compiles the hierarchy of files in a directory into a single source corpus. For example, it can gather all the files
into a single `.tex` file ready for compilation.
