def cprint(text, color):
    """
    Set the color of the text for command line output.

    Args:
        text (str): The text to be colored. It can contain placeholders for formatting.
        color (str): The color code. Available colors are:
            - 'red'
            - 'green'
            - 'yellow'
            - 'blue'
            - 'magenta'
            - 'cyan'
            - 'white'

    Returns:
        str: The colored text.
    """
    colors = {
        "r": "\033[91m",
        "g": "\033[92m",
        "y": "\033[93m",
        "b": "\033[94m",
        "m": "\033[95m",
        "c": "\033[96m",
        "w": "\033[97m",
        "reset": "\033[0m"
    }

    if color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)
    return 

def ctext(text, color):
    """
    Set the color of the text for command line output.

    Args:
        text (str): The text to be colored. It can contain placeholders for formatting.
        color (str): The color code. Available colors are:
            - 'red'
            - 'green'
            - 'yellow'
            - 'blue'
            - 'magenta'
            - 'cyan'
            - 'white'

    Returns:
        str: The colored text.
    """
    colors = {
        "r": "\033[91m",
        "g": "\033[92m",
        "y": "\033[93m",
        "b": "\033[94m",
        "m": "\033[95m",
        "c": "\033[96m",
        "w": "\033[97m",
        "reset": "\033[0m"
    }

    if color in colors:
        return f"{colors[color]}{text}{colors['reset']}"
    else:
        return text
     