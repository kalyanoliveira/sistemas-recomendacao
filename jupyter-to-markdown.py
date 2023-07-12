import os
import nbformat
from nbconvert import MarkdownExporter

jupyter_file_dir = os.path.join("filtragem-colaborativa")
jupyter_file_name = "surprise-CF"
jupyter_file_path = os.path.join(jupyter_file_dir, jupyter_file_name + ".ipynb")

with open(jupyter_file_path, 'r') as f:
    notebook = nbformat.read(f, as_version=4)

markdown_exporter = MarkdownExporter()
markdown_output, _ = markdown_exporter.from_notebook_node(notebook)

md_file_path = os.path.join(jupyter_file_dir, "to-markdown-" + jupyter_file_name + ".md")

with open(md_file_path, 'w') as f:
    f.write(markdown_output)
