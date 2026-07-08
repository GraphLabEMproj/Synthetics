import numpy as np

#from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle
from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.section import Section
from Synthetic3D.src.hard.structure.vector import Vector #, vector_to_int
from Synthetic3D.src.hard.vector_operation import random_dir_change
from Synthetic3D.src.organells.abstract_organell import Organell
from Synthetic3D.src.hard.split_curve import vector_sum_with_save_len
from Synthetic3D.src.hard.random_params import get_rand_int, choise_use_color_by_param, color_dim_check, get_color_index_fun_by_param
from Synthetic3D.src.utilities.check_of_params import check_param, update_param
from Synthetic3D.src.hard.structure.shells import FrameShell, ThickShell
#from Synthetic3D.src.hard.split_and_partition import partition_of_triangle
from Synthetic3D.src.hard.shell_expansion import shell_expansion

from Synthetic3D.src.hard.drawing_and_filliing.draw_cylinder import fill_small_capsule #, draw_small_cylinder_with_filling

from Synthetic3D.src.hard.drawing_and_filliing.fill_closed_shell import fill_closed_shell, get_valid_shell_points, fill_closed_shell_range_color
#from Synthetic3D.src.hard.structure.simple_tubular_cristae import create_tubular_cristae
from Synthetic3D.src.hard.structure.simple_tubular_cristae_fast import create_tubular_cristae_fast



from Synthetic3D.src.utilities.logging_config import logger

class Mitohondrion(Organell):
    def __init__(self, params=None):
        super().__init__()
        if params is not None and "mitohondrion" in params:
            self.params.update(params["mitohondrion"])
            self.comment = "Mitochondrion by params"
        else:
            self.comment = "Default mitochondrion"
        self.warnings = self._check_and_set_default_params()

        self.cristae = None

        ######################################################################################################################## PARAM !
        self.num_partition_of_triangles = self.params["num_partition_of_triangles"]
        self.warnings += self._Create()

    def _check_and_set_default_params(self) -> list[str]:
        warning_list = []

        warning_list+=check_param(self.params, "num_partition_of_triangles", 2)
        warning_list+=check_param(self.params, "mito_membrane_color", (255, 0, 0))
        warning_list+=check_param(self.params, "matrix_color", (0, 255, 0))
        warning_list+=check_param(self.params, "membrane_thickness", 2)
        warning_list+=check_param(self.params, "radius_of_section", 15)
        warning_list+=check_param(self.params, "twist_angle_of_sections", 0)
        warning_list+=check_param(self.params, "len_of_mitohondrion", (150, 250))


        warning_list+=check_param(self.params, "cristae_radius", (2, 6))
        warning_list+=check_param(self.params, "cristae_membrane_color", (0, 0, 255))
        warning_list+=check_param(self.params, "cristae_color", (0, 255, 255))
        warning_list+=check_param(self.params, "cristae_thickness", 2)

        #warning_list+=check_param(self.gen_params, "rv", 15)
        #warning_list+=check_param(self.gen_params, "rv", 15)

        if len(warning_list) != 0:
            logger.config(f'\tWarning Mitohondrion!\n{"\n".join(warning_list)}')
            warning_list = ["Warning Mitohondrion!"] + warning_list
        return warning_list

    def update_draw_config(self, config):
        new_params = config.get("mitohondrion", None)
        if new_params is not None:
            update_param(self.params, new_params, "mito_membrane_color")
            update_param(self.params, new_params, "matrix_color")
            update_param(self.params, new_params, "cristae_membrane_color")
            update_param(self.params, new_params, "cristae_color")
            update_param(self.params, new_params, "cristae_thickness")

    @staticmethod
    def make_TrickShell_from_shell_and_thickness(shell:FrameShell, thickness:float):
        up_thickness_shell = shell_expansion(shell, thickness)
        down_thickness_shell = shell_expansion(shell, -thickness)
        dict_of_vertexes = {
            "external": up_thickness_shell,
            "interior": down_thickness_shell
        }

        return ThickShell(dict_of_vertexes,
                          shell.get_frames())

    def _Create(self): # генерирует форму
        shell = self._Create_Shell()

        if isinstance(self.params["membrane_thickness"], (list, tuple)):
            self.params["membrane_thickness"] = get_rand_int(self.params["membrane_thickness"])

        self.shell = self.make_TrickShell_from_shell_and_thickness(shell, self.params["membrane_thickness"])
        self.shell.Partition_of_triangles(self.num_partition_of_triangles)
        self._CalculateViewData()

        return []

    def _Create_Shell(self)->FrameShell:
        shell = FrameShell()

        # митохондрия строится вертикально вверх и вниз
        position = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)  # направление -> z

        shell.add_frame_point(Vector(0, 0, 0))

        ############################################################################################# PARAM
        ############################################################################################# PARAM
        work_ru = get_rand_int(self.params["radius_of_section"])
        work_rv = get_rand_int(self.params["radius_of_section"])

        first_section = Section(position,
                                direction,
                                work_ru,
                                work_rv,
                                angle=self.params["twist_angle_of_sections"])

        self.section_max_radius_list = [first_section.max_radius]

        self.section_list = [first_section] ############################################################################### нужен для отладки

        # точки первой секции имеют индексы 0, 1, 2, 3
        shell.add_vertex_list(first_section.contour_list)
        # ребра первой секции имеют индексы 0, 1, 2, 3
        shell.add_edges(Edge(0, 1), # 1-3
                        Edge(1, 2), # 2-3
                        Edge(2, 3), # 1-4
                        Edge(3, 0)) # 2-4

        ############################################################################################# PARAM
        mito_len = get_rand_int(self.params["len_of_mitohondrion"])

        min_section_distance = max(work_ru, work_rv)

        self.count_of_section = 1

        len_of_sections = []

        for i in [1, -1]:  # построение в одном и обратном направлении
            len_of_sections.append(0)
            position = Vector(0, 0, 0)
            direction = Vector(0, 0, i)

            if i == 1:
                # строим по направлению первого сегмента
                last_section_indexes = [[0, 1, 2, 3], [0, 1, 2, 3]] # vertex and edge
            else:
                # reverve orientation
                # переворачиваем структуру, в этом случае первый сегмент становится последним, что позволит присоединить
                # обратное направление к нему, создавая непрерывное последовательное хранение структуры поверхности от
                # начала до конца митохондрии: конец_плюс -> центр_митохондрии -> конец_минус.
                shell.reverse()
                self.section_list.reverse()
                self.section_max_radius_list.reverse()

                # Порядок следования отличается, так как первая инверсная секция и начальная повернуты на 180 градусов.
                now_vertex_index = shell.get_vertex_count()
                now_edge_index = shell.get_edge_count()
                last_section_indexes = [[now_vertex_index-1,
                                         now_vertex_index-2,
                                         now_vertex_index-3,
                                         now_vertex_index-4],

                                        [now_edge_index-1,
                                         now_edge_index-2,
                                         now_edge_index-3,
                                         now_edge_index-4]]

            section_counter = 0
            while True: # выход через break по пределу длины
                section_distance = np.random.randint(min_section_distance*2, min_section_distance * 2+1) #####################################

                len_of_sections[-1] += section_distance

                # тут код вычисляющий точку, которая задается вектором от прошлой секции position в направлении direction
                # со случайным смещением в плоскости нормальной к direction
                change_value = np.random.random()*0.75  # значение от 0 до 0.75
                new_dir = random_dir_change(direction, change_value)
                result_point = position + direction * section_distance

                if (mito_len // 2) < (len_of_sections[-1]):
                    shell.add_frame_point(result_point)
                    Section.AddEdgeEndPoint(last_section_indexes, Vertex(result_point, direction/len_of_sections[-1]), shell)
                    break
                else:
                    # под конец уменьшить в 2 раза, но не меньше 5
                    new_section_ru = max(int(work_ru*(1 - (len_of_sections[-1]/(2 * mito_len)))),5)
                    new_section_rv = max(int(work_rv*(1 - (len_of_sections[-1]/(2 * mito_len)))),5)

                    new_section = Section(result_point,
                                          vector_sum_with_save_len(direction, new_dir),
                                          new_section_ru,
                                          new_section_rv,
                                          angle=self.params["twist_angle_of_sections"],
                                          reversed=i,
                                          )

                    direction = new_dir
                    position = result_point

                    # NEW SECTION INDEXES
                    new_v_index = shell.get_vertex_count()
                    shell.add_vertex_list(new_section.contour_list)
                    new_v_i_1 = new_v_index
                    new_v_i_2 = new_v_index + 1
                    new_v_i_3 = new_v_index + 2
                    new_v_i_4 = new_v_index + 3

                    new_ed_index = shell.get_edge_count()
                    ed1 = Edge(new_v_i_1, new_v_i_2)
                    ed2 = Edge(new_v_i_2, new_v_i_3)
                    ed3 = Edge(new_v_i_3, new_v_i_4)
                    ed4 = Edge(new_v_i_4, new_v_i_1)
                    shell.add_edges(ed1, ed2, ed3, ed4)

                    new_e_i_1 = new_ed_index
                    new_e_i_2 = new_ed_index + 1
                    new_e_i_3 = new_ed_index + 2
                    new_e_i_4 = new_ed_index + 3

                    new_section_indexes = [[new_v_i_1, new_v_i_2, new_v_i_3, new_v_i_4],
                                           [new_e_i_1, new_e_i_2, new_e_i_3, new_e_i_4]]

                    Section.AddSection(last_section_indexes, new_section_indexes, shell)
                    section_counter+=1
                    last_section_indexes = new_section_indexes
                    shell.add_frame_point(position)
                    self.section_list.append(new_section)
                    self.section_max_radius_list.append(new_section.max_radius)

            self.count_of_section += section_counter
        # подсчет что получилось
        self.mito_len = sum(len_of_sections)
        return shell

    def _Create_tubular_cristae(self, shell):
        ################################################################################################################### PARAMS ######################
        cristae_frame_list, cristae_radius_list = create_tubular_cristae_fast(shell.get_frames(),
                                                                         self.section_list,
                                                                         cristae_step=10,
                                                                         cristae_radius_param=self.params["cristae_radius"],
                                                                         cristae_gap=1,
                                                                         cristae_angle_deviation=30,
                                                                         density_cristae=0.25,
                                                                         max_count_added_cristae=1000,
                                                                         max_count_added_cristae_continue=1000,
                                                                         angle_by_frame_dir=45,
                                                                         overlap_radius=self.params["membrane_thickness"])

        return cristae_frame_list, cristae_radius_list

    def DrawCristae(self, data, constraint_shell, frames):
        logger.draw("\t\tcreate cristae")

        if self.cristae is None:
            cristae_frame_list, cristae_radius_list = create_tubular_cristae_fast(frames,
                                                                                  self.section_list,
                                                                                  cristae_step=10,
                                                                                  cristae_radius_param=self.params["cristae_radius"],
                                                                                  cristae_gap=1,
                                                                                  cristae_angle_deviation=30,
                                                                                  density_cristae=0.80,
                                                                                  max_count_added_cristae=1000,
                                                                                  max_count_added_cristae_continue=1000,
                                                                                  angle_by_frame_dir=30,
                                                                                  overlap_radius=self.params["membrane_thickness"])
            self.cristae = (cristae_frame_list, cristae_radius_list)

        logger.draw("\t\tdraw cristae in copy data")
        data_draw = data.copy()
        mask_shell_draw = np.zeros(data.shape[:3], np.uint8)
        fill_closed_shell(data=mask_shell_draw,
                          shell=constraint_shell,
                          color=1)

        for frame_of_crista, radiuce_cristae in zip(*self.cristae):
            len_frame = len(frame_of_crista)
            if len_frame < 2:
                continue
            else:
                # рисование внешки
                for i in range(len_frame - 1):
                    start_cylinder = frame_of_crista[i]
                    dir_cylinder = frame_of_crista[i + 1] - start_cylinder

                    fill_small_capsule(data_draw,
                                       start_cylinder,
                                       dir_cylinder,
                                       radiuce_cristae,
                                       color_dim_check(choise_use_color_by_param(self.params["cristae_membrane_color"]), data.shape))

        for frame_of_crista, radiuce_cristae in zip(*self.cristae):
            len_frame = len(frame_of_crista)
            if len_frame < 2:
                continue
            else:
                # рисование внутренности
                for i in range(len_frame - 1):
                    start_cylinder = frame_of_crista[i]
                    dir_cylinder = frame_of_crista[i + 1] - start_cylinder

                    fill_small_capsule(data_draw,
                                       start_cylinder,
                                       dir_cylinder,
                                       (radiuce_cristae - get_rand_int(self.params["cristae_thickness"]) - (radiuce_cristae//5)), # для самых больших ещё толще
                                       color_dim_check(choise_use_color_by_param(self.params["cristae_color"]), data.shape))


        logger.draw("\t\tdraw cristae into data using shell mask")
        data[mask_shell_draw[:,:,:]==1] = data_draw[mask_shell_draw[:,:,:]==1]


    def Draw(self, data):
        #logger.draw("MITO DRAW DON'T IMPLEMENTATION")
        #self.view_shell.Partition_of_triangles_while_len_of_edge_triangle_more_then_value(2, data.shape)
        self.view_shell.transform_coords2int()

        logger.draw("Start draw Mitohondrion")
        logger.draw("\tdraw outer")
        membrane_color_param = self.params["mito_membrane_color"]

        if get_color_index_fun_by_param(membrane_color_param) == 2:################################################################### задание основного и цвета в диапазоне для повышения разнообразия
            main_membrane_color = choise_use_color_by_param(membrane_color_param)
            min_range = membrane_color_param[0] - membrane_color_param[1]
            max_range = membrane_color_param[0] + membrane_color_param[1]
            change_3sigma = min(abs(main_membrane_color - min_range),
                                abs(max_range - main_membrane_color))
            use_membrane_color_param = (main_membrane_color, change_3sigma)
        else:
            use_membrane_color_param = membrane_color_param

        fill_closed_shell_range_color(data=data,
                                      shell=self.view_shell.dict_of_shells["external"],
                                      color_param=use_membrane_color_param)

        logger.draw("\tdraw inner")
        interior_color_param = self.params["matrix_color"]

        if get_color_index_fun_by_param(interior_color_param) == 2:################################################################### задание основного и цвета в диапазоне для повышения разнообразия
            main_interior_color = choise_use_color_by_param(interior_color_param)
            min_range = interior_color_param[0] - interior_color_param[1]
            max_range = interior_color_param[0] + interior_color_param[1]
            change_3sigma = min(abs(main_interior_color - min_range),
                                abs(max_range - main_interior_color))
            use_interior_color_param = (main_interior_color, change_3sigma)
        else:
            use_interior_color_param = interior_color_param

        fill_closed_shell_range_color(data=data,
                                      shell=self.view_shell.dict_of_shells["interior"],
                                      color_param=use_interior_color_param)

        logger.draw("\tdraw cristae")
        self.DrawCristae(data, self.view_shell.dict_of_shells["interior"], self.view_shell.get_frames())
        logger.draw("End draw Mitohondrion")

    def DrawMask(self, mask_data, color=None):
        #logger.draw("MITO MASK DON'T IMPLEMENTATION")

        #self.view_shell.Partition_of_triangles_while_len_of_edge_triangle_more_then_value(2, mask_data.shape)
        self.view_shell.transform_coords2int()

        work_color = self.params["mask_color"] if color is None else color
        work_color = color_dim_check(work_color, mask_data.shape)

        logger.draw("Start draw mask Mitohondrion")
        fill_closed_shell(data=mask_data,
                          shell=self.view_shell.dict_of_shells["external"],
                          color=work_color)
        logger.draw("End draw mask Mitohondrion")

    def DrawArea(self, cell_data, color) -> list[Vector]:
        #logger.draw("MITO DRAWAREA IS BAD IMPLEMENTATION")
        #self.Partition_of_triangles(4)

        #self.view_shell.Partition_of_triangles_while_len_of_edge_triangle_more_then_value(2, cell_data.shape)
        self.view_shell.transform_coords2int()

        #work_color = color_dim_check(work_color, mask_data.shape)
        logger.draw("Start draw area Mitohondrion")
        fill_closed_shell(data=cell_data,
                          shell=self.view_shell.dict_of_shells["external"],
                          color=color)
        valid_shell_points = get_valid_shell_points(cell_data, self.view_shell.dict_of_shells["external"])
        logger.draw(f"\tCreated {valid_shell_points.shape[0]} points to start expansion")
        logger.draw("End draw area Mitohondrion")
        return valid_shell_points
