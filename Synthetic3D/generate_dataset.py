import json
import time
from Synthetic3D.src.container.main_field import MainField
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.utilities.logging_config import clear_log_file

#Удалить прошлую сессию
clear_log_file()

log_list = []
start_time = time.time()
s_t_load = time.time()
# Открываем конфиг файл для чтения
with open("EPFL.json", "r", encoding="utf-8") as f:
    # Загружаем данные из файла в переменную
    config = json.load(f)
    print(config)
e_t_load = time.time()
log_list.append(f"Затраты времени на чтение файла {e_t_load-s_t_load} сек")
print(log_list[-1])

s_t_init = time.time()
generator = MainField(config)
e_t_init = time.time()
log_list.append(f"Затраты времени на инициализацию главного класса {e_t_init-s_t_init} сек")
print(log_list[-1])

s_t_init_data = time.time()
generator.CreateData()
e_t_init_data = time.time()
log_list.append(f"Затраты времени на создание массивов под данные {e_t_init_data-s_t_init_data} сек")
print(log_list[-1])

s_t_add = time.time()
generator.AddCellsByConfig()
e_t_add = time.time()
log_list.append(f"Затраты времени на создание добавление 16 органелл {e_t_add-s_t_add} сек")
print(log_list[-1])

s_t_exp = time.time()
generator.ExpansionOfRegions()
e_t_exp = time.time()
log_list.append(f"Затраты времени на расширение регионов { e_t_exp-s_t_exp} сек")
print(log_list[-1])

s_t_bg = time.time()
generator.CreateDataBackGround()
e_t_bg = time.time()
log_list.append(f"Затраты времени на заполнение фона датасета {e_t_bg-s_t_bg} сек")
print(log_list[-1])

s_t_dr = time.time()
generator.DrawDataset()
e_t_dr = time.time()
log_list.append(f"Затраты времени на рисование датасета {e_t_dr-s_t_dr} сек")
print(log_list[-1])

s_t_blur = time.time()
generator.AddBlur()
e_t_blur = time.time()
log_list.append(f"Затраты времени на размытие датасета {e_t_blur-s_t_blur} сек")
print(log_list[-1])

s_t_noise = time.time()
generator.AddNoise()
e_t_noise = time.time()
log_list.append(f"Затраты времени на зашумление датасета {e_t_noise-s_t_noise} сек")
print(log_list[-1])

s_t_dr_mask = time.time()
generator.DrawMasks()
e_t_dr_mask = time.time()
log_list.append(f"Затраты времени на рисование масок датасета {e_t_dr_mask-s_t_dr_mask} сек")
print(log_list[-1])

s_t_w = time.time()
generator.WriteDataset(log_list)
e_t_w = time.time()
log_list.append(f"Затраты времени на сохранение датасета {e_t_w-s_t_w} сек")
print(log_list[-1])

end_time = time.time()
log_list.append(f"Времени на генерацию {end_time-start_time} сек")
print(log_list[-1])

print()
for str_log in log_list:
    print(str_log)

view_vtk_3D_data(generator.data)
view_vtk_3D_data(generator.masks[0])
view_vtk_3D_data(generator.masks[1])
view_vtk_3D_data(generator.masks[2])
