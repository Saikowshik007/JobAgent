from typing import Union, List, Generator, Optional
import config


def read_jobfile(filename: str) -> str:
    """
    Reads the content of a job file.

    Args:
        filename (str): Path to the job file.

    Returns:
        str: Content of the job file.
    """
    try:
        with open(filename, "r") as stream:
            return stream.read().strip()
    except OSError as e:
        config.getLogger("file_handler").error(f"The {filename} could not be read.")
        raise e


def generator_key_in_nested_dict(
        keys: Union[str, List[str]], nested_dict: dict
) -> Generator:
    """
    Generates values for specified keys in a nested dictionary.

    Args:
        keys (Union[str, List[str]]): Key or list of keys to search for.
        nested_dict (dict): The nested dictionary to search in.

    Yields:
        Generator: Values corresponding to the specified keys.
    """
    if hasattr(nested_dict, "items"):
        for key, value in nested_dict.items():
            if (isinstance(keys, list) and key in keys) or key == keys:
                yield value
            if isinstance(value, dict):
                # Don't use utils. prefix for a function in the same module
                yield from generator_key_in_nested_dict(keys, value)
            elif isinstance(value, list):
                for item in value:
                    # Don't use utils. prefix for a function in the same module
                    yield from generator_key_in_nested_dict(keys, item)


def get_dict_field(field: str, data_dict: dict) -> Optional[dict]:
    """
    Retrieves a field from a dictionary.

    Args:
        field (str): The field to retrieve.
        data_dict (dict): The dictionary to retrieve the field from.

    Returns:
        Optional[dict]: The value of the field, or None if the field is missing.
    """
    try:
        return data_dict[field]
    except KeyError as e:
        message = f"`{field}` is missing in raw resume."
        config.getLogger("file_handler").warning(message)
    return None