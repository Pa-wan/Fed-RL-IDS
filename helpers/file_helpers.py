

def save_array_to_file(file_name: str, items: list):
    """
        Save a list of items to a file.
    """
    with open(file_name, "w") as f:
        for item in items:
            f.write(str(item) + ", ")


def read_list_from_file(file_name: str):
    """
        Read a list from a file.
    """
    with open(file_name, "r") as f:
        return f.read().split(", ")
