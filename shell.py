import basic2
import re
import traceback
import sys

def collect_input():
    """Collect multi-line input until a complete statement is recognized."""
    lines = []
    prompt = 'basic > '
    while True:
        try:
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
                if re.search(r'\b(if|loop|while|method)\b', l) and not re.search(r'\bend\b', l):
                    block_level += 1

                # Detect continuation of if block with elif or else
                elif re.search(r'\b(elif|else)\b', l) and not re.search(r'\bend\b', l):
                    # These don't increase the block level as they're part of an existing if
                    pass

                # Detect block closing with 'end' keyword (possibly followed by the block type)
                if re.search(r'\bend\b', l):
                    block_level -= 1
            
            # If all blocks are closed, we can execute
            if block_level <= 0:
                return current_text

            # Otherwise continue collecting input with a different prompt
            prompt = '... '
        except KeyboardInterrupt:
            print("\nInput interrupted")
            return ""
        except EOFError:
            print("\nEOF detected")
            return ""

def main():
    print("BASIC Programming Language - Enhanced Shell")
    print("Type 'exit()' to exit the shell")
    
    while True:
        try:
            text = collect_input()
            if text.strip() == "":
                continue
                
            if text.strip() == "exit()":
                print("Exiting...")
                break

            result, error = basic2.run('<stdin>', text)

            if error:
                print("\n" + "=" * 40)
                print("ERROR DETECTED:")
                print("=" * 40)
                print(error.as_string())
                print("=" * 40 + "\n")
            elif result:
                if hasattr(result, 'elements') and len(result.elements) == 1:
                    print(repr(result.elements[0]))
                else:
                    print(repr(result))
        except KeyboardInterrupt:
            print("\nExecution interrupted. Returning to shell...")
        except Exception as e:
            print("\n" + "=" * 40)
            print("SHELL ERROR (not a language error):")
            print("=" * 40)
            print(f"Type: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print("\nPython Traceback:")
            traceback.print_exc()
            print("=" * 40 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)