import re


def find_all_boxed_expression_contents(content: str) -> list[str]:
    """Return a list of all boxed expression contents in the given content."""
    boxed_expression_contents = re.findall("\\\\boxed\\{(.*?)\\}", content)
    return boxed_expression_contents


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
