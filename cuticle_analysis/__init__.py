
import logging

from rich.logging import RichHandler

from .cli import cli

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


def main():
    cli()
