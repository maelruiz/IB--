def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    
    # Extract line number for display
    line_num = pos_start.ln + 1
    
    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result with line number
        result += f'{line_num+i}: {line}\n'
        result += '   ' + ' ' * col_start + '^' * max(1, col_end - col_start) + ' '
        
        # Add the specific error position indicator
        if i == 0:
            result += f'column {col_start+1}'
        result += '\n'

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '    ')  # Replace tabs with 4 spaces for clearer display
