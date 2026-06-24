import cv2
import os
from PIL import Image

def write_dataset(data, path_to_save, name_of_dir, save_channel=None, save_as_gif=False):
    if not os.path.isdir(path_to_save):
        print(f"Создаю дирректорию '{path_to_save}' со всеми промежуточными")
        os.makedirs(path_to_save)

    all_path = os.path.join(path_to_save, name_of_dir)
    if not os.path.isdir(all_path):
        print(f"Создаю дирректорию '{name_of_dir}' в {path_to_save}")
        os.mkdir(all_path)

    if save_as_gif:
        images = []

    shape_of_frame_data = data.shape[:3]
    for z in range(shape_of_frame_data[0]):
        slice = data[z, :, :, :] if save_channel is None else data[z, :, :, save_channel]
        cv2.imwrite(f"{all_path}/synthetic_data_{shape_of_frame_data[2]}_{shape_of_frame_data[1]}_{z}.png",
                    slice)

        if save_as_gif:
            img = Image.fromarray(slice)
            images.append(img)

    if save_as_gif:
        images[0].save(f'{path_to_save}/{name_of_dir}.gif', save_all=True, append_images=images[1:], loop=0, duration=200)



def write_datamask(data, path_to_save, name_of_dir):
    if not os.path.isdir(path_to_save):
        print(f"Создаю дирректорию '{path_to_save}' со всеми промежуточными")
        os.makedirs(path_to_save)

    all_path = os.path.join(path_to_save, name_of_dir)
    if not os.path.isdir(all_path):
        print(f"Создаю дирректорию '{name_of_dir}' в {path_to_save}")
        os.mkdir(all_path)

    shape_of_frame_data = data.shape[:3]
    for z in range(shape_of_frame_data[0]):
        slice = data[z, :, :]
        cv2.imwrite(f"{all_path}/synthetic_data_{shape_of_frame_data[2]}_{shape_of_frame_data[1]}_{z}.png",
                    slice)
