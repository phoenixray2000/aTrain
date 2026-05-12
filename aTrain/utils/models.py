import os
import traceback
import urllib.error
import urllib.request
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Manager
from pathlib import Path

from aTrain_core.settings import load_languages
from aTrain_core.globals import MODELS_DIR, REQUIRED_MODELS, REQUIRED_MODELS_DIR
from aTrain_core.load_resources import get_model, load_model_config_file, remove_model
from nicegui import run, ui
from nicegui.run import SubprocessException
from nicegui.run import setup as setup_process_pool

from aTrain.components.dialogs.download import close_dialog_download, dialog_download
from aTrain.components.dialogs.error import dialog_error


def read_downloaded_models() -> list:
    directories_to_search = [MODELS_DIR, REQUIRED_MODELS_DIR]
    model_names = set(load_model_config_file().keys())
    all_downloaded_models = []

    for directory in directories_to_search:
        os.makedirs(directory, exist_ok=True)
        all_file_directories = [
            dir_entry.name for dir_entry in os.scandir(directory) if dir_entry.is_dir()
        ]
        all_file_directories.sort(reverse=True)

        for directory_name in all_file_directories:
            directory_path = Path(directory) / directory_name
            # Some repositories keep model binaries in nested folders.
            if directory_name in model_names and any(directory_path.rglob("*.bin")):
                all_downloaded_models.append(directory_name)

    return all_downloaded_models


def read_transcription_models() -> list:
    all_models = read_downloaded_models()
    while "diarize" in all_models:
        all_models.remove("diarize")
    return all_models


def read_model_metadata() -> list:
    model_metadata = load_model_config_file()
    all_models = list(model_metadata.keys())
    downloaded_models = read_downloaded_models()
    all_models_metadata = []

    for model in all_models:
        config = model_metadata[model]
        required = model in REQUIRED_MODELS
        storage_dir = REQUIRED_MODELS_DIR if required else MODELS_DIR
        revision = config["revision"]
        repo_id = config["repo_id"]
        model_info = {
            "model": model,
            "size": config["repo_size_human"],
            "downloaded": model in downloaded_models,
            "required": required,
            "repo_id": repo_id,
            "revision": revision,
            "download_url": f"https://huggingface.co/{repo_id}/tree/{revision}",
            "storage_path": str(storage_dir / model),
        }
        all_models_metadata.append(model_info)

    all_models_metadata = sorted(
        all_models_metadata,
        key=lambda x: (not x["required"], not x["downloaded"], x["model"]),
    )

    return all_models_metadata


def model_languages(model: str) -> dict:
    languages_dict = load_languages()
    all_models_configs = load_model_config_file()
    model_config: dict = all_models_configs[model]
    if "languages" in model_config.keys():
        languages: list = model_config["languages"]
        languages_dict = {language: languages_dict[language] for language in languages}
    return languages_dict


async def download_model(model: str):
    with Manager() as manager:
        progress = manager.dict({"current": 0, "total": 999999})
        dialog_download(progress, model)
        try:
            check_internet()
            await run.cpu_bound(get_model, model=model, progress=progress)
            close_dialog_download()
            ui.navigate.reload()

        except BrokenProcessPool:
            remove_model(model)
            setup_process_pool()
            close_dialog_download()
            ui.navigate.reload()

        except SubprocessException as e:
            remove_model(model)
            close_dialog_download()
            dialog_error(error=e.original_message, traceback=e.original_traceback)

        except Exception as e:
            remove_model(model)
            close_dialog_download()
            dialog_error(str(e), traceback.format_exc())


def check_internet() -> None:
    """A function to check whether the user is connected to the internet."""
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
    except urllib.error.URLError:
        raise ConnectionError(
            "We cannot reach Hugging Face. Most likely you are not connected to the internet."
        )
