import re

def check_balance(filename):
    with open(filename, 'r') as f:
        content = f.read()

    stack = []
    lines = content.splitlines()
    
    # Simple state machine to ignore strings and comments
    in_string = False
    string_char = ''
    in_comment = False # //
    in_multiline_comment = False # /* */
    
    for i, line in enumerate(lines):
        j = 0
        while j < len(line):
            char = line[j]
            
            # Handle comments and strings
            if in_comment:
                if j == len(line) - 1: # End of line resets single line comment
                    in_comment = False
                j += 1
                continue
                
            if in_multiline_comment:
                if char == '*' and j + 1 < len(line) and line[j+1] == '/':
                    in_multiline_comment = False
                    j += 1
                j += 1
                continue
                
            if in_string:
                if char == '\\': # Escape
                    j += 1
                elif char == string_char:
                    in_string = False
                j += 1
                continue
                
            # Start of string/comment?
            if char in ["'", '"', '`']:
                in_string = True
                string_char = char
                j += 1
                continue
                
            if char == '/' and j + 1 < len(line):
                if line[j+1] == '/':
                    in_comment = True
                    j += 1
                    continue
                elif line[j+1] == '*':
                    in_multiline_comment = True
                    j += 1
                    continue
            
            # Check brackets
            if char in ['{', '(', '[']:
                stack.append((char, i + 1, j + 1))
            elif char in ['}', ')', ']']:
                if not stack:
                    print(f"Error: Unexpected '{char}' at line {i+1} col {j+1}")
                    return
                
                last_char, last_line, last_col = stack.pop()
                expected = {'{': '}', '(': ')', '[': ']'}[last_char]
                if char != expected:
                    print(f"Error: Mismatched '{char}' at line {i+1} col {j+1}. Expected '{expected}' closing '{last_char}' from line {last_line} col {last_col}")
                    return
            
            j += 1
        
        # Reset single line comment at end of line (already handled, but just safety)
        in_comment = False

    if stack:
        first_unclosed = stack[0]
        print(f"Error: Unclosed '{first_unclosed[0]}' at line {first_unclosed[1]} col {first_unclosed[2]}")
        print(f"Total unclosed: {len(stack)}")
        print("Last few unclosed:", stack[-5:])
    else:
        print("No unbalanced brackets found.")

check_balance('src/App.jsx')
