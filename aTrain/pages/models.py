from nicegui import ui

from aTrain.layouts.base import base_layout
from aTrain.utils.models import download_model, read_model_metadata, remove_model


@ui.page("/models")
def page():
    models = read_model_metadata()
    with base_layout():
        ui.label("Model Manager").classes("text-lg text-dark font-bold")
        with ui.list().classes("w-full").props("separator"):
            with ui.item():
                with ui.grid(columns="minmax(0, 44px) 1.1fr 0.7fr 1.7fr 1fr") as grid:
                    grid.classes("w-full text-grey text-xs items-end")
                    ui.label("#")
                    ui.label("Model")
                    ui.label("Download Size")
                    ui.label("Storage Location")
                    ui.label("Actions")
            for i, model in enumerate(models):
                with ui.item().classes("hover:bg-gray-100"):
                    with ui.grid(
                        columns="minmax(0, 44px) 1.1fr 0.7fr 1.7fr 1fr"
                    ) as grid:
                        grid.classes("w-full items-center gap-x-4")
                        ui.label(str(i + 1)).classes("font-light")
                        with ui.column().classes("gap-1 min-w-0"):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(model["model"]).classes("font-medium")
                                if model["required"]:
                                    ui.badge("Required", color="dark")
                            ui.label(model["repo_id"]).classes(
                                "font-light text-xs text-grey break-all"
                            )
                        ui.label(model["size"]).classes("font-light")
                        ui.label(model["storage_path"]).classes(
                            "font-mono text-xs font-light break-all"
                        )
                        with ui.row():
                            with ui.link(target=model["download_url"], new_tab=True):
                                btn_link = ui.button("Link", color="gray-100")
                                btn_link.props(
                                    "no-caps size=0.7rem unelevated text-color=dark icon=open_in_new"
                                )
                            if model["downloaded"]:
                                if model["required"]:
                                    ui.badge("Installed", color="green")
                                else:
                                    btn_delete = ui.button("Delete", color="gray-100")
                                    btn_delete.props("no-caps size=0.7rem unelevated")
                                    btn_delete.on_click(
                                        lambda m=model: (
                                            remove_model(m["model"]),
                                            ui.navigate.reload(),
                                        )
                                    )
                            else:
                                btn_download = ui.button("Download", color="dark")
                                btn_download.props("no-caps size=0.7rem unelevated")
                                btn_download.on_click(
                                    lambda m=model: download_model(m["model"])
                                )
