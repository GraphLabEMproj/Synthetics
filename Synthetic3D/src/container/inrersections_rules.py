import numpy as np
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.organells.vesicles import Vesicles
from Synthetic3D.src.organells.mitohondrion import Mitohondrion
from Synthetic3D.src.hard.dictance import distance_point_to_segment, min_distance_between_segments

# Константы расширения нужны чтобы корректно работал алгоритм разрастания
EXPECTITION_TWO_EMPTYORGANELLS = 10
EXPECTITION_EMPTYORGANELLS_AND_VESICLES = 10
EXPECTITION_TWO_VESICLES = 10
EXPECTITION_EMPTYORGANELLS_AND_MITOHONDRION = 10
EXPECTITION_VESICLES_AND_MITOHONDRION = 10
EXPECTITION_TWO_MITOHONDRIA = 20


def CheckRoughDistanceEstimateIntersectionWithVesicles(pos, gap_dist, vesicle: Vesicles):
    if np.linalg.norm(pos - vesicle.position) < gap_dist + vesicle.max_cloud_radius + vesicle.params["thickness"] + vesicle.max_radius:
        return True
    else:
        return False

def CheckIntersectionTwoEmptyOrganelle(org1: EmptyOrganelle, org2: EmptyOrganelle):
    pos1 = org1.position
    pos2 = org2.position
    if np.linalg.norm(pos1 - pos2) < org1.radius + org2.radius + EXPECTITION_TWO_EMPTYORGANELLS:
        return True
    else:
        return False

def CheckIntersectionEmptyOrganelleAndVesicles(org1: EmptyOrganelle, org2: Vesicles):
    pos1 = org1.position
    if CheckRoughDistanceEstimateIntersectionWithVesicles(pos1, org1.radius+EXPECTITION_EMPTYORGANELLS_AND_VESICLES, org2):
        for i, pos2 in enumerate(org2.view_shell.get_frames()):
            radius_ves = org2.radius_list[i] + org2.params["thickness"]
            if np.linalg.norm(pos1 - pos2) < org1.radius + radius_ves + EXPECTITION_EMPTYORGANELLS_AND_VESICLES:
                return True
    return False

def CheckRoughDistanceEstimateIntersectionTwoVesicles(org1: Vesicles, org2: Vesicles):
    if np.linalg.norm(org1.position - org2.position) <\
        org1.max_cloud_radius + org2.max_cloud_radius +\
        org1.params["thickness"]     + org2.params["thickness"] +\
        org1.max_radius     + org2.max_radius+\
        EXPECTITION_TWO_VESICLES:
        return True
    else:
        return False

def CheckIntersectionTwoVesicles(org1: Vesicles, org2: Vesicles):
    if CheckRoughDistanceEstimateIntersectionTwoVesicles(org1, org2):
        for i, pos1 in enumerate(org1.view_shell.get_frames()):
            radius_ves1 = org1.radius_list[i] + org1.params["thickness"]
            for j, pos2 in enumerate(org2.view_shell.get_frames()):
                radius_ves2 = org2.radius_list[j] + org2.params["thickness"]
                if np.linalg.norm(pos1 - pos2) < radius_ves1 + radius_ves2 + EXPECTITION_TWO_VESICLES:
                    return True
    return False


def CheckIntersectionMitohondriaAndEmptyOrganelle(org1: Mitohondrion, org2: EmptyOrganelle):
    pos_empty = org2.position
    radius_empty = org2.radius

    # Проверка удаленности от первой внутренней точки фрейма
    dist_mito_from_start_to_frame = np.linalg.norm(org1.view_shell.get_frame(0) - org1.view_shell.get_frame(1)) + \
                                    org1.params["thickness"]
    if np.linalg.norm(pos_empty - org1.view_shell.get_frame(1)) < \
            radius_empty + dist_mito_from_start_to_frame + EXPECTITION_EMPTYORGANELLS_AND_MITOHONDRION:
        return True
    else:
        # Проверка удаленности от последней внутренней точки фрейма
        dist_mito_from_end_to_frame = np.linalg.norm(org1.view_shell.get_frame(-1) - org1.view_shell.get_frame(-2)) + \
                                      org1.params["thickness"]

        if np.linalg.norm(pos_empty - org1.view_shell.get_frame(-2)) < \
                radius_empty + dist_mito_from_end_to_frame + EXPECTITION_EMPTYORGANELLS_AND_MITOHONDRION:
            return True
        else:
            # Проверка удаленности от сегментов фрейма
            # начало и конец тоже в фреймах и точек на 1 больше чем сегментов (-2 -1)
            len_of_mito_segments = len(org1.view_shell.get_frames()) - 3
            for i in range(len_of_mito_segments):
                start_point_frame_segment = org1.view_shell.get_frame(i + 1)
                end_point_frame_segment = org1.view_shell.get_frame(i + 2)

                max_radius_beetween_sections = max(org1.section_max_radius_list[i],
                                                   org1.section_max_radius_list[i + 1]) + \
                                               org1.params["thickness"]

                if distance_point_to_segment(pos_empty, start_point_frame_segment, end_point_frame_segment) < \
                        max_radius_beetween_sections + radius_empty + EXPECTITION_EMPTYORGANELLS_AND_MITOHONDRION:
                    return True
                else:
                    continue
            return False


def CheckIntersectionMitohondriaAndVesicles(org1: Mitohondrion, org2: Vesicles):
    # Проверка удаленности от первой внутренней точки фрейма
    dist_mito_from_start_to_frame = np.linalg.norm(org1.view_shell.get_frame(0) - org1.view_shell.get_frame(1)) + \
                                    org1.params["membrane_thickness"] + EXPECTITION_VESICLES_AND_MITOHONDRION

    if CheckRoughDistanceEstimateIntersectionWithVesicles(org1.view_shell.get_frame(1), dist_mito_from_start_to_frame, org2):
        for i, pos_ves in enumerate(org2.view_shell.get_frames()):
            radius_ves = org2.radius_list[i] + org2.params["thickness"]
            if np.linalg.norm(pos_ves - org1.view_shell.get_frame(1)) < \
                    radius_ves + dist_mito_from_start_to_frame:
                return True

    # Проверка удаленности от последней внутренней точки фрейма
    dist_mito_from_end_to_frame = np.linalg.norm(org1.view_shell.get_frame(-1) - org1.view_shell.get_frame(-2)) + \
                                      org1.params["membrane_thickness"] + EXPECTITION_VESICLES_AND_MITOHONDRION
    if CheckRoughDistanceEstimateIntersectionWithVesicles(org1.view_shell.get_frame(-2), dist_mito_from_end_to_frame, org2):
        for i, pos_ves in enumerate(org2.view_shell.get_frames()):
            radius_ves = org2.radius_list[i] + org2.params["thickness"]

            if np.linalg.norm(radius_ves - org1.view_shell.get_frame(-2)) < \
                    radius_ves + dist_mito_from_end_to_frame:
                return True

    # Проверка удаленности от сегментов фрейма
    # начало и конец тоже в фреймах и точек на 1 больше чем сегментов (-2 -1)
    len_of_mito_segments = len(org1.view_shell.get_frames()) - 3
    for i in range(len_of_mito_segments):
        start_point_frame_segment = org1.view_shell.get_frame(i + 1)
        end_point_frame_segment = org1.view_shell.get_frame(i + 2)

        max_radius_beetween_sections = max(org1.section_max_radius_list[i],
                                           org1.section_max_radius_list[i + 1]) + \
                                       org1.params["membrane_thickness"] + EXPECTITION_VESICLES_AND_MITOHONDRION

        if distance_point_to_segment(org2.position, start_point_frame_segment, end_point_frame_segment) <\
            max_radius_beetween_sections + org2.max_cloud_radius + org2.params["thickness"] + org2.max_radius:

            for i, pos_ves in enumerate(org2.view_shell.get_frames()):
                radius_ves = org2.radius_list[i] + org2.params["thickness"]

                if distance_point_to_segment(pos_ves, start_point_frame_segment, end_point_frame_segment) < \
                        max_radius_beetween_sections + radius_ves:
                    return True
    return False

def CheckIntersectionTwoMitohondria(org1: Mitohondrion, org2: Mitohondrion):
    # Проверка концов митохондрий

    union_membrane_thickness = org1.params["membrane_thickness"] + org2.params["membrane_thickness"]

    # вычисление дистанции от конца до ближайших внутренних точек фрейма + толщина оболочки
    dist_mito_from_start_to_frame_1 = np.linalg.norm(org1.view_shell.get_frame(0) - org1.view_shell.get_frame(1)) + \
                                      union_membrane_thickness
    dist_mito_from_start_to_frame_2 = np.linalg.norm(org2.view_shell.get_frame(0) - org2.view_shell.get_frame(1)) + \
                                      union_membrane_thickness

    dist_mito_from_end_to_frame_1 = np.linalg.norm(org1.view_shell.get_frame(-1) - org1.view_shell.get_frame(-2)) + \
                                      union_membrane_thickness
    dist_mito_from_end_to_frame_2 = np.linalg.norm(org2.view_shell.get_frame(-1) - org2.view_shell.get_frame(-2)) + \
                                      union_membrane_thickness

    # вычисление проверка что краевые области не пересекаются
    if np.linalg.norm(org1.view_shell.get_frame(1) - org2.view_shell.get_frame(1)) < \
            dist_mito_from_start_to_frame_1 + dist_mito_from_start_to_frame_2 + EXPECTITION_TWO_MITOHONDRIA:
        return True
    if np.linalg.norm(org1.view_shell.get_frame(1) - org2.view_shell.get_frame(-2)) < \
            dist_mito_from_start_to_frame_1 + dist_mito_from_end_to_frame_2 + EXPECTITION_TWO_MITOHONDRIA:
        return True
    if np.linalg.norm(org1.view_shell.get_frame(-2) - org2.view_shell.get_frame(1)) < \
            dist_mito_from_end_to_frame_1 + dist_mito_from_start_to_frame_2 + EXPECTITION_TWO_MITOHONDRIA:
        return True
    if np.linalg.norm(org1.view_shell.get_frame(-2) - org2.view_shell.get_frame(-2)) < \
            dist_mito_from_end_to_frame_1 + dist_mito_from_end_to_frame_2 + EXPECTITION_TWO_MITOHONDRIA:
        return True

    # Проверка удаленности сегментов митохондрий
    # Проверка каждого с каждым, так как могут пересекаться только 2 сегмента (скрещиванием)
    len_of_mito_segments_1 = len(org1.view_shell.get_frames()) - 3
    len_of_mito_segments_2 = len(org2.view_shell.get_frames()) - 3
    for i in range(len_of_mito_segments_1):
        start_point_frame_segment_1 = org1.view_shell.get_frame(i + 1)
        end_point_frame_segment_1 = org1.view_shell.get_frame(i + 2)

        max_radius_beetween_sections_1 = max(org1.section_max_radius_list[i],
                                             org1.section_max_radius_list[i + 1]) + \
                                         union_membrane_thickness + EXPECTITION_TWO_MITOHONDRIA

        for i in range(len_of_mito_segments_2):
            start_point_frame_segment_2 = org2.view_shell.get_frame(i + 1)
            end_point_frame_segment_2 = org2.view_shell.get_frame(i + 2)

            max_radius_beetween_sections_2 = max(org2.section_max_radius_list[i],
                                                 org2.section_max_radius_list[i + 1]) + \
                                             union_membrane_thickness

            # вычисление минимальной длины двух отрезков
            if min_distance_between_segments(start_point_frame_segment_1, end_point_frame_segment_1,
                                             start_point_frame_segment_2, end_point_frame_segment_2) < \
                    max_radius_beetween_sections_1 + max_radius_beetween_sections_2:
                return True

    return False


def CheckTwoOrganells(org1, org2):

    if isinstance(org1, EmptyOrganelle):
        if isinstance(org2, EmptyOrganelle):
            return CheckIntersectionTwoEmptyOrganelle(org1, org2)
        elif isinstance(org2, Vesicles):
            return CheckIntersectionEmptyOrganelleAndVesicles(org1, org2)
        elif isinstance(org2, Mitohondrion):
            return CheckIntersectionMitohondriaAndEmptyOrganelle(org2, org1)
        else:
            raise NotImplementedError(
                "At the moment, EmptyOrganelle can only be tested with Vesicles, EmptyOrganelle and Mitohondrion class")

    elif isinstance(org1, Vesicles):
        if isinstance(org2, Vesicles):
            return CheckIntersectionTwoVesicles(org1, org2)
        elif isinstance(org2, EmptyOrganelle):
            return CheckIntersectionEmptyOrganelleAndVesicles(org2, org1)
        elif isinstance(org2, Mitohondrion):
            return CheckIntersectionMitohondriaAndVesicles(org2, org1)
        else:
            raise NotImplementedError(
                "At the moment, Vesicle/es can only be tested with Vesicles, EmptyOrganelle and Mitohondrion class")

    elif isinstance(org1, Mitohondrion):
        if isinstance(org2, Vesicles):
            return CheckIntersectionMitohondriaAndVesicles(org1, org2)
        elif isinstance(org2, EmptyOrganelle):
            return CheckIntersectionMitohondriaAndEmptyOrganelle(org1, org2)
        elif isinstance(org2, Mitohondrion):
            return CheckIntersectionTwoMitohondria(org1, org2)
        else:
            raise NotImplementedError(
                "At the moment, Mitohondrion can only be tested with Vesicles, EmptyOrganelle and Mitohondrion class")

    else:
        raise NotImplementedError(
            f"At the moment, {type(org1)} can't be tested.")
