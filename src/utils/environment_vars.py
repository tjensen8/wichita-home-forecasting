"""
Gets the API key from environment.
"""

from dotenv import load_dotenv
import yaml
import os


def get_bls_key():
    get_environment_variables()
    return os.getenv("BLS_KEY")


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
    with open(dir, "r") as f:
        yaml_f = yaml.safe_load(f)
    return yaml_f
