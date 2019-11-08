import importlib
import sys

import lazy_object_proxy


def load_settings():
    """
    A function which loads settings from settings.py.
    :return:
    """
    settings_path = "scFNN.general.settings"
    try:
        loaded_settings = importlib.import_module(settings_path)
        # sys.stderr.write("Settings loaded successfully. \n")
    except ImportError as e:
        sys.stderr.write("Failed to load settings. \n")
        raise e

    return loaded_settings


# Load settings once
settings = lazy_object_proxy.Proxy(load_settings)
