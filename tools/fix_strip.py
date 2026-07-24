with open('crates/ui/src/paint.rs', 'r') as f:
    lines = f.readlines()

# 1. Remove hover overlay block (lines 371-383 approximately)
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == '// Hover highlight — only on interactive elements':
        # Skip until next non-empty non-brace line
        i += 1
        while i < len(lines) and not (lines[i].strip().startswith('// `background-image`')):
            i += 1
        new_lines.append(lines[i])
        i += 1
        continue
    # 2. Remove custom arrow block
    if line.strip() == '// Draw .arrow chevrons (original uses ::before which we don\'t support).':
        i += 1
        while i < len(lines) and not (lines[i].strip().startswith('// Collect children sorted by z-index.')):
            i += 1
        new_lines.append(lines[i])
        i += 1
        continue
    new_lines.append(line)
    i += 1

# 3. Remove is_interactive function at the end
content = ''.join(new_lines)
idx = content.find('/// Returns true for elements that should show a hover highlight.')
if idx != -1:
    content = content[:idx].rstrip() + '\n'

with open('crates/ui/src/paint.rs', 'w') as f:
    f.write(content)

print('Removed hover overlay, custom arrows, is_interactive')
