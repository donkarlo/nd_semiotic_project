# Goal

The goal of this project is to support everything related to semiotics, with natural language as its most important
branch. It provides a façade (a shared interface) for using different language models, embeddings, and vector databases.

It prioritizes the hierarchy of corpora in the file system.

For files such as YAML, JSON, etc., it considers and prioritizes the data entry hierarchies within these files.

# RAG
The RAG system uses the given set of folder paths to build chunks and embeddings, and then uses an arbitrary LLM to
answer users' prompts within the directory system, while considering the hierarchy of the files and their contents.

# Agentic Rag
This sub package is responsible to assign goals to members to try until they achieve it. 

# Vector databases
Provides an interface to work with different vector databases without having to learn each of their unique interface

# Language Conversion

It develops conversion tools between different languages and formats, such as translating from YAML to LaTeX files.

# Compiling

It compiles the hierarchy of files in a directory into a single source corpus. For example, it can gather all the files
into a single `.tex` file ready for compilation.
