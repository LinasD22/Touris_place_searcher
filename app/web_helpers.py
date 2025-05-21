def get_markdown(text: str) -> str:
    lines = text.split('\n')
    new_markdown = ''

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('==') and stripped.endswith('=='):
            header = stripped.lstrip('=').rstrip('=').strip()
            if header and i + 1 < len(lines) and lines[i+1].strip():
                new_markdown += f"## {header}\n"
        else:
            new_markdown += line + '\n'

    return new_markdown