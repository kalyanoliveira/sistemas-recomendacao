import re
import os

md_file_dir = os.path.join("filtragem-colaborativa")
md_file_name = "to-markdown-surprise-CF copy"
md_file_extension = ".md"
md_file_path = os.path.join(md_file_dir, md_file_name + md_file_extension)

with open(md_file_path, "r") as f_md:
    lines = [line for line in f_md.readlines()]

for index, line in enumerate(lines):
    if found := re.search(pattern=r'^# (.*)', string=line):
        if re.search(pattern=r'^# %%', string=line):
            if re.search(pattern=r"^# %% \[markdown\]", string=line):
                lines[index] = ""
            else:
                lines[index] = "```python\n```\n"
        else:
            lines[index] = found.group(1) + "\n"
    else:
        continue 

with open(os.path.join(md_file_dir, md_file_name + "-processed" + md_file_extension), "w") as f_md:
    f_md.writelines(lines)