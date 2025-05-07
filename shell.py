import basic2
import re

def collect_input():
    """Collect multi-line input until a complete statement is recognized."""
    lines = []
    prompt = 'basic > '
    while True:
        line = input(prompt)
        lines.append(line)

        # Try to parse the current input
        current_text = '\n'.join(lines)

        # If empty, just return
        if current_text.strip() == "":
            return current_text

        # Check for complete code by looking at block structure
        block_level = 0
        text_lines = current_text.split('\n')

        for l in text_lines:
            # Detect block opening
            if re.search(r'\b(if|loop|for|while|method)\b', l) and 'end' not in l:
                block_level += 1

            # Detect block closing with 'end' keyword
            if 'end' in l:
                block_level -= 1
        
        # If all blocks are closed, we can execute
        if block_level <= 0:
            return current_text

        # Otherwise continue collecting input with a different prompt
        prompt = '... '

while True:
    try:
        text = collect_input()
        if text.strip() == "":
            continue

        result, error = basic2.run('<stdin>', text)

        if error:
            print(error.as_string())
        elif result:
            if hasattr(result, 'elements') and len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(repr(result))
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Shell error: {e}")

