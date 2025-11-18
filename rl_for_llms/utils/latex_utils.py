def find_all_boxed_expression_contents(text: str) -> list[str]:
    """Find all contents of boxed expressions in the given text."""
    out = []
    search_text = "\\boxed{"
    i = 0
    while True:
        i = text.find(search_text, i)
        if i < 0:
            break
        i += len(search_text)
        depth = 0
        start = i
        j = start
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    out.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
            j += 1
        else:
            break
    return out



def get_boxed_expression(expression: str) -> str:
    """Return the expression wrapped in a boxed if it is not already boxed."""
    boxed_expression_contents = find_all_boxed_expression_contents(expression)
    if len(boxed_expression_contents) == 0:
        boxed_expression = f"\\boxed{{{expression}}}"
    else:
        boxed_expression = expression
    return boxed_expression


def get_last_boxed_expression(content: str) -> str:
    """Return the content of the last boxed expression in the given content."""
    boxed_expression_contents = find_all_boxed_expression_contents(content)
    if len(boxed_expression_contents) == 0:
        return get_boxed_expression(content)
    last_boxed_expression_content = boxed_expression_contents[-1]
    boxed_expression = get_boxed_expression(last_boxed_expression_content)
    return boxed_expression
