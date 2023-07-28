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

    md_file_path = os.path.join(markdown_files_dir, "to-md-" + jupyter_file_name + ".md")

    with open(md_file_path, 'w') as f:
        f.write(markdown_output)

    # with open(md_file_path, 'r') as f:
    #     lines = f.readlines()

    #     for index in range(len(lines) - 2):
    #         content = lines[index] + lines[index + 1] + lines[index + 2]
    #         p = r'```\n\n    +.'
    #         if re.search(pattern=p, string=content):
    #             lines[index + 1] = "\n    OUTPUT\n"

    #     for index in range(len(lines) - 5):
    #         content = lines[index] + lines[index + 1] + lines[index + 2] + lines[index + 3] + lines[index + 4] + lines[index + 5]
    #         p = r'```\n\n\n\n\n    +.'
    #         if re.search(pattern=p, string=content):
    #             lines[index + 4] = "\n    OUTPUT\n"

    # with open(md_file_path, "w") as f:
    #     f.writelines(lines)