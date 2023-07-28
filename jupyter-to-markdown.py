import re
import os
import nbformat
from nbconvert import MarkdownExporter

jupyter_files_dir = "jupyter-nbs"
markdown_files_dir = "markdowns"

for jupyter_file in os.listdir(jupyter_files_dir):

    jupyter_file_name = jupyter_file.split(".")[0]

    jupyter_file_path = os.path.join(jupyter_files_dir, jupyter_file_name + ".ipynb")

    with open(jupyter_file_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)

    markdown_exporter = MarkdownExporter()
    markdown_output, _ = markdown_exporter.from_notebook_node(notebook)

    md_file_path = os.path.join(markdown_files_dir, jupyter_file_name + ".md")

    with open(md_file_path, 'w') as f:
        f.write(markdown_output)