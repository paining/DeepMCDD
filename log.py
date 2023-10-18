import logging
import logging.config
import yaml
import os
import sys

def set_logger(config_file:str, result_path:str, skip_unknown_exception:bool=False):
    try:
        with open(config_file, "r") as f:
            dict = yaml.safe_load(f)
        os.makedirs(result_path, exist_ok=True)
        for k, v in dict["handlers"].items():
            if "filename" in v:
                v["filename"] = os.path.join(result_path, v["filename"])
        logging.config.dictConfig(dict)

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception
    except FileNotFoundError as e:
        print("Cannot find config file:", config_file)
    except Exception as e:
        print("Exception Occur:", e)