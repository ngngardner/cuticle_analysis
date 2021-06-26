
import click

from .core.app import start
from .core.dataset import dataset_setup, unzip_dataset, download_dataset, download_unzip
from .gui.gui import application


@click.group()
def cli():
    pass


cli.add_command(start)
cli.add_command(dataset_setup)
cli.add_command(download_dataset)
cli.add_command(unzip_dataset)
cli.add_command(download_unzip)

# gui
cli.add_command(application)
