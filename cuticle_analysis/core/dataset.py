
import logging
import os

import click
from rich import print

logger = logging.getLogger(__name__)


@click.command()
@click.pass_context
def download_unzip(ctx):
    ctx.invoke(download_dataset)
    ctx.invoke(unzip_dataset)


@click.command()
def download_dataset():
    import gdown # type: ignore

    print("Downloading dataset...")
    url = "https://drive.google.com/uc?id=1xABlWE790uWmT0mMxbDjn9KZlNUB6Y57"
    output = "./dataset.zip"
    gdown.download(url, output, quiet=False)

    assert os.path.isfile(output)
    print("Downloaded dataset!")


@click.command()
def unzip_dataset():
    print("Extracting dataset...")
    zip_path = f"./dataset.zip"
    folder_path = f"./dataset"

    try:
        if not os.path.isdir(folder_path):
            if os.path.isfile(zip_path):
                from zipfile import PyZipFile
                zf = PyZipFile(zip_path)
                zf.extractall(path=folder_path)

                assert os.path.isdir(folder_path)
                print("Extracted dataset!")
            else:
                raise Exception(f"Failed to find {zip_path}.")
        else:
            raise Exception(f"Directory {folder_path} already exists.")

    except Exception as e:
        logger.error(f'Failed to unzip dataset:" {e}')
        raise e


menu = {
    "0": "Quit",
    "1": "Download dataset.zip from Google Drive",
    "2": "I already have dataset.zip"
}

actions = {
    "0": SystemExit,
    "1": download_unzip,
    "2": unzip_dataset
}


@click.command()
@click.pass_context
def dataset_setup(ctx):
    if os.path.isdir("dataset"):
        print("Found dataset!")
        return
    else:
        for key in menu:
            print(f'{key}: {menu[key]}')
        dataset_init = click.prompt(
            "How should dataset be initialized?", type=str)
        func = actions.get(dataset_init, None)
        ctx.invoke(func)

    assert(os.path.isdir("dataset"))
    print("Successfully initialized dataset.")
