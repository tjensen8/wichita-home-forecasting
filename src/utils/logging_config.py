import logging
from environment_vars import get_environment_variables


def set_logging_config():
    logging.basicConfig(
        filename=get_environment_variables["LOGGING_LOCATION"],
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    log_announce = "Logging Configuration Set."

    print(log_announce)
    logging.info(log_announce)


if __name__ == "__main__":
    set_logging_config()
