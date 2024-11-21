import os
import re
import yaml


def mkdir(path):
    """
    It creates the path (directory) if it doesn't exist

    :param path: the path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_project_root():
    """
    It returns the path to the project root

    :return: The path to the root of the project.
    """
    utils_path = os.path.dirname(os.path.abspath(__file__))
    root_end_idx = list(re.finditer("src", utils_path))[-1].start()
    root_path = utils_path[0:root_end_idx]
    return root_path


def flatten_dict(d):
    """
    It flattens a dictionary

    :param d: the dictionary to be flattened

    :return: the flattened dictionary
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def get_config():
    """
    It reads the config.yaml file and returns it as a dictionary

    :return: the config dictionary.
    """
    root_dir = get_project_root()
    with open(root_dir + 'config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config["project_root"] = root_dir

    # Turn relative project paths into absolute paths
    for key in config["project_paths"].keys():
        config["project_paths"][key] = \
            os.path.join(root_dir, config["project_paths"][key])
        # Create the directory if it doesn't exist
        mkdir(config["project_paths"][key])

    # Turn relative data paths into absolute paths
    for key in config["data_paths"].keys():
        config["data_paths"][key] = \
            os.path.join(root_dir, config["data_paths"][key])
        
    config["default_params"] = flatten_dict(config["default_params"])

    return config

