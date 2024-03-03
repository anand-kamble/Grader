""" This module contains utility functions for the project. """
import os

def list_files(directory):
    """
    List all files in the specified directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of file names in the directory.
    """
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files
