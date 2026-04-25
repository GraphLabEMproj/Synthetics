from enum import Enum
import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

class CristaeSegmentType(Enum):
    REGIONAL = 0
    SEGMENT = 1
    NODAL = 2
    START = 3
    END = 4

class CristaSegment:
    """
    Класс для итеративного наращивания крист вдоль митохондрии. Кристы создаются с состоянием REGIONAL. При начале
    распространения им ставится статус START как начальная точка и END как завершающая. active_status указывает что
    данную ноду нужно проверить в алгоритме распростронения или пропустить как неактивную.
    """
    def __init__(self,
                 position: Vector,
                 cristae_type:CristaeSegmentType = CristaeSegmentType.SEGMENT,
                 prev_neighbor: 'CristaSegment' = None
                 ):
        self.active_status = True
        self.cristae_type = cristae_type
        self.prev_neighbor = prev_neighbor
        self.next_neighbor = None
        self.position = position

    def to_nodal_transform(self):
        if self.cristae_type is CristaeSegmentType.SEGMENT:
            if self.next_neighbor is None:
                self.next_neighbor = []
            else:
                self.next_neighbor = [self.next_neighbor]

            if self.prev_neighbor is None:
                self.prev_neighbor = []
            else:
                self.prev_neighbor = [self.prev_neighbor]

            self.cristae_type = CristaeSegmentType.NODAL
        elif self.cristae_type in {CristaeSegmentType.REGIONAL,
                                   CristaeSegmentType.START,
                                   CristaeSegmentType.END}:
            raise RuntimeError(f"Нельзя сделать из краевой ноды {self.cristae_type} узловую !")

        else:
            pass # уже узловая

    @staticmethod
    def off_active_if_not_node(segment):
        if segment.cristae_type is not CristaeSegmentType.NODAL:
            segment.active_status = False

    def added_last_segment(self, new_cristae: 'CristaSegment'):
        if self.cristae_type is CristaeSegmentType.NODAL:
            self.prev_neighbor.append(new_cristae)
        elif self.prev_neighbor is None and self.cristae_type is not CristaeSegmentType.START:
            self.prev_neighbor = new_cristae
            self.off_active_if_not_node(new_cristae)
            if self.cristae_type is CristaeSegmentType.REGIONAL:
                self.cristae_type = CristaeSegmentType.END
                self.active_status = False
        else:
            if self.prev_neighbor is not None:
                raise RuntimeError("Предыдущая нода уже установлена")
            else:
                raise RuntimeError("Нельзя установить предыдущий узел к начальной ноде!")

    def added_next_segment(self, new_cristae: 'CristaSegment'):
        if self.cristae_type is CristaeSegmentType.NODAL:
            self.next_neighbor.append(new_cristae)
        elif self.next_neighbor is None and self.cristae_type is not CristaeSegmentType.END:
            self.next_neighbor = new_cristae
            self.off_active_if_not_node(self)
            if self.cristae_type is CristaeSegmentType.REGIONAL:
                self.cristae_type = CristaeSegmentType.START
        else:
            if self.next_neighbor is not None:
                raise RuntimeError("Следующая нода уже установлена")
            else:
                raise RuntimeError("Нельзя установить следующий узел к конечному!")
        new_cristae.added_last_segment(self)

    def added_segments_in_node(self, new_cristae: 'CristaSegment'|list['CristaSegment']):
        # добавляет к текущей новые как следующие, и превращась в ноду
        self.to_nodal_transform()
        if isinstance(new_cristae, CristaSegment):
            self.next_neighbor.append(new_cristae)
        else:
            self.next_neighbor += new_cristae

    def ended_segment(self, new_cristae:'CristaSegment'):
        # добавляет к текущей новую как предыдущую, завершая её и превращаясь в ноду
        self.to_nodal_transform()
        self.prev_neighbor.append(new_cristae)
        self.off_active_if_not_node(new_cristae)

    def get_type(self):
        return self.cristae_type


def init_tubular_cristae(count_of_start_cristae, fun_of_gen_new_points):
    cristae_start_list = []

    list_of_start_segment_points = []
    for i in range(count_of_start_cristae):
        point = fun_of_gen_new_points(list_of_start_segment_points)
        if point is not None:
            cristae_start_list.append(CristaSegment(point, CristaeSegmentType.START))
    return cristae_start_list

def search_and_pop_closest_free_segment(check_segment, list_of_reserv) -> CristaSegment:
    assert len(list_of_reserv) > 0, "Нужно из чего-то искать"
    check_seg_pos = check_segment.position
    min_dist = np.linalg.norm(check_seg_pos-list_of_reserv[0].position)
    min_index = 0
    for i, segment in enumerate(list_of_reserv):
        dist = np.linalg.norm(check_seg_pos-segment.position)
        if dist < min_dist:
            min_index = i
            min_dist = dist
    return list_of_reserv.pop(min_index)

def search_closest_segment(check_segment, list_of_segments) -> int:
    assert len(list_of_segments) > 0, "Нужно из чего-то искать"
    check_seg_pos = check_segment.position
    min_index = -1
    min_dist = None

    for i, segment in enumerate(list_of_segments):
        if segment.cristae_type in {CristaeSegmentType.SEGMENT,
                                    CristaeSegmentType.NODAL,
                                    CristaeSegmentType.REGIONAL}:

            dist = np.linalg.norm(check_seg_pos - segment.position)
            if min_index < 0 or dist < min_dist:
                min_dist = dist
                min_index = i
    return min_index

def check_of_intersection(crist_segment, closest_segment):
    return False

def finished_tubular_cristae(active_cristae_list, fun_of_gen_new_segments, list_of_reserv_end_segment, segments_list):
    fun_of_gen_new_segments(len(active_cristae_list), list_of_reserv_end_segment)
    for crist_segment in active_cristae_list:
        free_closest_segment = search_and_pop_closest_free_segment(crist_segment, list_of_reserv_end_segment)

        if check_of_intersection(crist_segment, free_closest_segment):   # Проверка перекается ли последний с остальными
            closest_segment_index = search_closest_segment(crist_segment, segments_list)
            if closest_segment_index == -1:
                raise RuntimeError("Невозможно закончить сегмент")
            else:
                segments_list[closest_segment_index].ended_segment(crist_segment) # слить текущий в ближайший
        else:
            free_closest_segment.cristae_type = CristaeSegmentType.REGIONAL
            crist_segment.added_last_segment(free_closest_segment)

def probapility():
    if np.random.random() < 0.5:
        return True
    else:
        return False

def continue_tubular_segments(active_cristae_list,
                              fun_of_gen_target_direction,  # смещает точку в полусфере направления фрейма
                              reserv_segments):
    new_active_list = []
    for cristae_segment in active_cristae_list:
        success = False
        new_segment = None

        if probapility(): # завершаем ближайшим
            new_segment = search_and_pop_closest_free_segment(cristae_segment, reserv_segments)
            if check_of_intersection(cristae_segment, new_segment):
                reserv_segments.append(new_segment) # вернуть обратно
            else:
                success = True

        if success is False:
            new_point = fun_of_gen_target_direction(cristae_segment.position)
            new_segment = CristaSegment(new_point, CristaeSegmentType.SEGMENT)

        if check_of_intersection(cristae_segment, new_segment):
            new_active_list.append(cristae_segment)
        else:
            cristae_segment.added_next_segment(new_segment)
            new_active_list.append(new_segment)

    return new_active_list











