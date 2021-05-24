from typing import List
from matplotlib import pyplot as plt
from CMRSegment.common.graph import Graph, Vertex
from CMRSegment.common.topology import PointSequence, Arc, Curve, Point, L2Distance, tangent, normal


class CardiacVertex(Vertex):
    def __init__(self, node_id: int, point: Point, label: str, tangent: Point, normal: Point, elastic_force: Point, stiff_force: Point):
        super().__init__(node_id)
        self.label = label
        self.point = point
        self.tangent = tangent
        self.normal = normal
        self.elastic_force = elastic_force
        self.stiff_force = stiff_force
        self.v_t = Point((0, 0))
        self.v_n = Point((0, 0))

    def __repr__(self):
        return f"uid: {self.uid()}, point: {self.point}"


class CardiacSnakeGraph(Graph):
    def __init__(self):
        super().__init__()
        self.lv_start_id = None
        self.lv_end_id = None

    @classmethod
    def from_curve(cls, curve: Curve, label: str):
        graph = cls()

        for idx, point in enumerate(curve):
            tangent = curve.tangent(point)
            normal = curve.normal(point)
            elastic_force = curve[idx + 1] - curve[idx] * 2 + curve[idx - 1]
            stiff_force = curve[idx + 2] * (-1) + curve[idx + 1] * 4 - curve[idx] * 6 + curve[idx - 1] * 4 - curve[idx - 2] * 2
            if idx == 0:
                prev_vertex = CardiacVertex(
                    idx, curve[idx], label, tangent, normal,
                    elastic_force=elastic_force,
                    stiff_force=stiff_force
                )
                graph.add_vertex(prev_vertex)
                continue
            vertex = CardiacVertex(
                idx, curve[idx], label, tangent, normal,
                elastic_force=elastic_force,
                stiff_force=stiff_force
            )
            graph.add_vertex(vertex)
            graph.add_edge(frm=prev_vertex, to=vertex, directed=True)
            prev_vertex = vertex
        graph.add_edge(frm=graph.get_vertex(graph.num_vertices - 1), to=graph.get_vertex(0), directed=True)
        return graph

    @classmethod
    def from_curves(cls, lv_curve: Curve, rv_arc: Arc, lv_connected_point1: Point, lv_connected_point2: Point):
        lv_connected_point1_id = lv_curve.index(lv_connected_point1)
        lv_connected_point2_id = lv_curve.index(lv_connected_point2)

        graph = cls.from_curve(lv_curve, label="lv")
        # connect connected points to new blue bdry
        dist_func = L2Distance()
        dist1 = dist_func(lv_connected_point1, rv_arc[0])
        dist2 = dist_func(lv_connected_point1, rv_arc[-1])

        if dist1 > dist2:
            lv_start_id = lv_connected_point2_id
            lv_end_id = lv_connected_point1_id
        else:
            lv_start_id = lv_connected_point1_id
            lv_end_id = lv_connected_point2_id
        from_vertex = graph.get_vertex(lv_start_id)
        start_id = graph.num_vertices
        for idx, point in enumerate(rv_arc):
            id = start_id + idx
            tangent = rv_arc.tangent(point)
            normal = rv_arc.normal(point)
            if idx == 0:
                elastic_force = rv_arc[idx + 1] - rv_arc[idx] * 2 + from_vertex.point
                stiff_force = rv_arc[idx + 2] * (-1) + rv_arc[idx + 1] * 4 - rv_arc[idx] * 5 + from_vertex.point * 2
                from_vertex.elastic_force += rv_arc[idx] - from_vertex.point
            elif idx == 1:
                elastic_force = rv_arc[idx + 1] - rv_arc[idx] * 2 + rv_arc[idx - 1]
                stiff_force = rv_arc[idx + 2] * (-1) + rv_arc[idx + 1] * 4 - rv_arc[idx] * 6 + rv_arc[idx - 1] * 4 - from_vertex.point * 2
            elif idx == len(rv_arc) - 1:
                elastic_force = rv_arc[idx - 1] - rv_arc[idx] * 2 + graph.get_vertex(lv_end_id).point
                stiff_force = (graph.get_vertex(lv_end_id).point * 2 - rv_arc[idx] * 5 + rv_arc[idx - 1] * 4 - rv_arc[idx - 2]) * 1
            elif idx == len(rv_arc) - 2:
                elastic_force = rv_arc[idx + 1] - rv_arc[idx] * 2 + rv_arc[idx - 1]
                stiff_force = graph.get_vertex(lv_end_id).point * (-1) + rv_arc[idx + 1] * 4 - rv_arc[idx] * 6 + rv_arc[idx - 1] * 4 - rv_arc[idx - 2] * 2
            else:
                elastic_force = rv_arc[idx + 1] - rv_arc[idx] * 2 + rv_arc[idx - 1]
                stiff_force = rv_arc[idx + 2] * (-1) + rv_arc[idx + 1] * 4 - rv_arc[idx] * 6 + rv_arc[idx - 1] * 4 - rv_arc[idx - 2] * 2
            vertex = CardiacVertex(
                id, point, label="rv", tangent=tangent, normal=normal,
                elastic_force=elastic_force,
                stiff_force=stiff_force
            )
            graph.add_edge(frm=from_vertex, to=vertex, directed=True)
            from_vertex = vertex
        graph.add_edge(frm=vertex, to=graph.get_vertex(lv_end_id), directed=True)
        graph.lv_start_id = lv_start_id
        graph.lv_end_id = lv_end_id
        print(lv_start_id, lv_end_id)
        return graph

    def plot(self, fig=None):
        if fig is None:
            fig = plt.figure()
        directions = list(self.traverse())
        prev_node = directions[0].src
        plt.plot(prev_node.point[1], prev_node.point[0], 'yo')
        for edge in directions:
            src = edge.src
            dst = edge.dst
            if dst.label == "rv" or src.label == "rv":
                color = "bx-"
            else:
                color = "rx-"
            plt.plot(
                [src.point[1], dst.point[1]],
                [src.point[0], dst.point[0]],
                color,
            )
            # plt.arrow(
            #     src.point[1], src.point[0],
            #     dst.point[1] - src.point[1], dst.point[0] - src.point[0], width=1
            # )
        # plt.plot(prev_node.point[1], prev_node.point[0], 'go')

        return fig

    def find_children(self, vertex: CardiacVertex, same_label: bool = False):
        edges = self.find_edges(vertex)
        if same_label:
            return [edge.dst for edge in edges if edge.dst.label == vertex.label]
        return [edge.dst for edge in edges]

    def find_parents(self, vertex: CardiacVertex, same_label: bool = False):
        if not same_label:
            parents = [edge.src for edges in self.edges.values() for edge in edges if edge.dst.uid() == vertex.uid()]
            return parents
        return [edge.src for edges in self.edges.values() for edge in edges if edge.dst.uid() == vertex.uid() and edge.src.label == vertex.label]

    def update_vectors(self):
        start_node = self.get_vertex(self.lv_start_id)
        children = self.find_children(start_node)
        # update tangents
        lv_child = [child for child in children if child.label == "lv"][0]
        rv_child = [child for child in children if child.label == "rv"][0]

        # lv path
        visited = []
        prev_node = start_node
        this_node = lv_child
        next_node = self.find_children(this_node)[0]
        while True:
            if this_node in visited:
                break
            node_tangent = tangent(prev_node.point, next_node.point)

            elastic_force = next_node.point - this_node.point * 2 + prev_node.point

            next_next_node = self.find_children(next_node)[0]
            prev_prev_node = self.find_parents(prev_node)[0]
            stiff_force = next_next_node.point * (-1) + next_node.point * 4 - this_node.point * 6 + prev_node.point * 4 - prev_prev_node.point

            this_node.elastic_force = elastic_force
            this_node.stiff_force = stiff_force

            this_node.tangent = node_tangent
            visited.append(this_node)
            prev_node = this_node
            this_node = next_node
            next_node = self.find_children(this_node)[0]

        # rv path
        this_node = rv_child
        visited = [this_node]
        next_node = self.find_children(this_node)[0]
        node_tangent = tangent(this_node.point, next_node.point)
        this_node.tangent = node_tangent
        elastic_force = next_node.point - this_node.point * 2 + start_node.point
        stiff_force = self.find_children(next_node)[0].point * (-1) + next_node.point * 4 - this_node.point * 5 + start_node.point * 2
        this_node.elastic_force = elastic_force
        this_node.stiff_force - stiff_force
        start_node.elastic_force += this_node.point - start_node.point

        prev_node = this_node
        this_node = next_node
        next_node = self.find_children(this_node)[0]
        while True:
            if this_node.label == "lv":
                break
            node_tangent = tangent(prev_node.point, next_node.point)
            this_node.tangent = node_tangent
            elastic_force = next_node.point - this_node.point * 2 + prev_node.point
            if next_node.label == "rv":
                next_next_node = self.find_children(next_node)[0]
                prev_prev_node = self.find_parents(prev_node)[0]
                stiff_force = next_next_node.point * (-1) + next_node.point * 4 - this_node.point * 6 + prev_node.point * 4 - prev_prev_node.point * 2
            else:
                prev_prev_node = self.find_parents(prev_node)[0]
                stiff_force = (next_node.point * 2 - this_node.point * 5 + prev_node.point * 4 - prev_prev_node.point) * 1

            this_node.elastic_force = elastic_force
            this_node.stiff_force = stiff_force
            visited.append(this_node)
            prev_node = this_node
            this_node = next_node
            next_node = self.find_children(this_node)[0]

        # update elastic force of lv end
        this_node.elastic_force += prev_node.point - this_node.point
        # update normal
        for child in children:
            if child.label == "lv":
                # lv path
                visited = []
                prev_node = self.get_vertex(start_node.uid())
                this_node = child
                next_node = self.find_children(this_node)[0]
                while True:
                    if this_node in visited:
                        break
                    node_normal = normal(prev_node.tangent, next_node.tangent)
                    this_node.normal = node_normal
                    visited.append(this_node)
                    prev_node = this_node
                    this_node = next_node
                    next_node = self.find_edges(this_node)[0].dst
            elif child.label == "rv":
                # rv path
                this_node = child
                visited = [this_node]
                next_node = self.find_children(this_node)[0]
                node_normal = normal(this_node.tangent, next_node.tangent)
                this_node.normal = node_normal
                prev_node = this_node
                this_node = next_node
                next_node = self.find_children(this_node)[0]
                while True:
                    if this_node.label == "lv":
                        break
                    node_normal = normal(prev_node.tangent, next_node.tangent)
                    this_node.normal = node_normal
                    visited.append(this_node)
                    prev_node = this_node
                    this_node = next_node
                    next_node = self.find_children(this_node)[0]
