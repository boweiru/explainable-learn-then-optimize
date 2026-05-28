import re

def remove_element_block(inp_path, output_path=None):
    if output_path is None:
        output_path = inp_path

    with open(inp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start_pattern = re.compile(r'^\s*\*ELEMENT\s*,.*TYPE\s*=\s*CPS3.*ELSET\s*=\s*misc1', re.IGNORECASE)
    end_pattern   = re.compile(r'^\s*\*ELEMENT\s*,.*TYPE\s*=\s*C3D8R.*ELSET\s*=\s*auto', re.IGNORECASE)
    new_lines = []
    in_block = False
    for line in lines:
        if start_pattern.match(line):
            in_block = True
            continue
        if end_pattern.match(line):
            in_block = False
            new_lines.append(line)
            continue
        if not in_block:
            new_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Removed block from '*ELEMENT...,ELSET=misc1' to (before) '*ELEMENT...,ELSET=auto'.")
    if output_path == inp_path:
        print(f"File updated in-place: {inp_path}")
    else:
        print(f"Saved to: {output_path}")

inp_file = "adaqusModel1.inp"
remove_element_block(inp_file)