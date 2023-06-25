"""
Gets the API key from environment.
"""

from dotenv import load_dotenv
import yaml
import os


def get_bls_key():
    return load_dotenv()["BLS_KEY"]


def get_environment_variables():
    return load_dotenv()


class ExceptionMessage(Exception):
    """Exception with a simple informative message."""

    pass


def load_yaml(dir: str):
    """Loads a YAML file from a specified directory.

    Args:
        dir (str): Directory.

    Returns:
        dict: YAML data as a dictionary.
    """
    if os.path.isdir(dir):
        with open(dir, "r") as f:
            yaml_f = yaml.safe_load()
        return yaml_f
    else:
        raise ExceptionMessage(f"Directory {dir} is not a directory.")
