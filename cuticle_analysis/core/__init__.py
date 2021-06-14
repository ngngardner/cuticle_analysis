import logging

from rich.logging import RichHandler

from .app import App


def init():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )


def start_app():
    init()
    app = App()
    app.start()
