import flet as ft
from flet import Column, ElevatedButton, Image, Text, Page, AppBar, View, colors, FilePickerUploadFile
from pathlib import Path
import numpy as np
import icons
from ultralytics import YOLO
import os
import shutil
import requests

# Установка FLET_SECRET_KEY, если он не установлен
if not os.getenv("FLET_SECRET_KEY"):
    os.environ["FLET_SECRET_KEY"] = os.urandom(12).hex()
def join_files(part_files, output_file):
    with open(output_file, 'wb') as output:
        for part_file in part_files:
            with open(part_file, 'rb') as pf:
                output.write(pf.read())
    print('Файлы склеены!')


def main(page: ft.Page):

    part_files = ['./assets/part-00-DermDiagnose_YOLOv8m_256x256_20_epoch_best.pt', './assets/part-01-DermDiagnose_YOLOv8m_256x256_20_epoch_best.pt']
    join_files(part_files, './assets/DermDiagnose_YOLOv8m_256x256_20_epoch_best.pt')

    page.title = "DermDiagnose"

    def image_processing(file_path):

        model = YOLO('./assets/DermDiagnose_YOLOv8m_256x256_20_epoch_best.pt')
        results = model(file_path, save_txt=False)

        top5_classes = results[0].probs.top5
        top5conf_classes = (results[0].probs.top5conf).tolist()
        top5_classes_names = [model.names[i] for i in top5_classes]
        first_cls.value = f'{top5_classes_names[0]}   {round(top5conf_classes[0] * 100, 2)}%'
        second_cls.value = f'{top5_classes_names[1]}   {round(top5conf_classes[1] * 100, 2)}%'
        third_cls.value = f'{top5_classes_names[2]}   {round(top5conf_classes[2] * 100, 2)}%'
        fourth_cls.value = f'{top5_classes_names[3]}   {round(top5conf_classes[3] * 100, 2)}%'
        fifth_cls.value = f'{top5_classes_names[4]}   {round(top5conf_classes[4] * 100, 2)}%'
        page.go("/result")

    def upload_files(e: ft.FilePickerResultEvent, selected_file: bool, selected_file_name: str):

        upload_folder = "./assets/uploads"
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
            print(f"Папка {upload_folder} удалена.")
        else:
            print(f"Папка {upload_folder} не существует.")

        upload_list = []
        if pick_files_dialog.result != None and pick_files_dialog.result.files != None:
            selected_file = True
            f = pick_files_dialog.result.files[0]
            selected_file_name = f.name
            upload_list.append(
                FilePickerUploadFile(
                    f.name,
                    upload_url=page.get_upload_url(f.name, 600),
                )
            )
            pick_files_dialog.upload(upload_list)
            print(f'{f.name} uploaded!')
            print(f'selected_file = {selected_file}')
        return selected_file, selected_file_name

    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_file = False
        selected_file_name = ""
        selected_file, selected_file_name = upload_files(e, selected_file, selected_file_name)
        print(f'selected_file = {selected_file}')
        if selected_file == True:
            page.go("/loading")
            image_processing(f'assets/uploads/{selected_file_name}')
            selected_file_path.value = f'./uploads/{selected_file_name}'
            print(selected_file_path.value)

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_file_path = ft.Text()
    first_cls = ft.Text(size=20)
    second_cls = ft.Text(size=20)
    third_cls = ft.Text(size=20)
    fourth_cls = ft.Text(size=20)
    fifth_cls = ft.Text(size=20)
    loading = ft.Text(size=20)
    loading.value = "Loading..."
    page.overlay.append(pick_files_dialog)

    def route_change(e):
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    Image(src="logo.jpg",
                            width=578,
                            height=291,
                            fit= 'fill',
                            border_radius=ft.border_radius.all(10)),
                    ElevatedButton("Choose image", icon=ft.icons.UPLOAD_FILE, color=colors.GREEN_400, on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False)),
                ],
                horizontal_alignment = ft.CrossAxisAlignment.CENTER,
                vertical_alignment = ft.MainAxisAlignment.CENTER
            )
        )
        if page.route == "/loading":
            page.views.append(
                View(
                    "/loading",
                    [
                        Image(src="logo.jpg",
                              width=578,
                              height=291,
                              fit='fill',
                              border_radius=ft.border_radius.all(10)),
                        loading
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    vertical_alignment=ft.MainAxisAlignment.CENTER
                )
            )
        if page.route == "/result":
            page.views.append(
                View(
                    "/result",
                    [
                        AppBar(title=Text("Result"), bgcolor=colors.GREEN_400),
                        Image(
                            src=selected_file_path.value,
                            width=300,
                            height=300,
                        ),
                        first_cls,
                        second_cls,
                        third_cls,
                        fourth_cls,
                        fifth_cls,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    vertical_alignment=ft.MainAxisAlignment.CENTER
                )
            )
        page.update()

    def view_pop(e):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    page.go(page.route)

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets", upload_dir="assets/uploads")#, view=ft.AppView.WEB_BROWSER