"""
ProcessGPT BPMN XML generator (backend-only, no frontend dependency).

This module is a Python port of `process-gpt-vue3/src/components/BPMNXmlGenerator.vue`
focused on the exact execution path used by `createBpmnXml(...)`:
  - auto layout (Graph + SugiyamaLayout) assigns node positions + edge waypoints + role boundaries
  - BPMN XML (MODEL + DI) is generated with the same ids/structure/rules

Important:
  - XML attribute ordering may differ from browser XMLSerializer, but element structure/content is preserved.
  - This code intentionally mirrors the original logic (including some quirks).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Namespaces (match BPMNXmlGenerator.vue)
# ---------------------------------------------------------------------------

NS_BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS_BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
NS_DC = "http://www.omg.org/spec/DD/20100524/DC"
NS_DI = "http://www.omg.org/spec/DD/20100524/DI"
NS_UENGINE = "http://uengine"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"

ET.register_namespace("xsi", NS_XSI)
ET.register_namespace("bpmn", NS_BPMN)
ET.register_namespace("bpmndi", NS_BPMNDI)
ET.register_namespace("uengine", NS_UENGINE)
ET.register_namespace("dc", NS_DC)
ET.register_namespace("di", NS_DI)


def _q(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _bool_str(v: Any) -> str:
    return "true" if bool(v) else "false"


# ---------------------------------------------------------------------------
# Auto layout (ported from src/components/autoLayout/graph-algorithm.js)
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    id: str
    label: str
    x: float = 0.0
    y: float = 0.0
    layer: int = 0
    order: int = 0
    group: Optional[str] = None
    width: float = 0.0
    height: float = 0.0
    nodeType: Optional[str] = None
    barycenter: float = 0.0


@dataclass
class _Edge:
    source: str
    target: str
    feedback: bool = False
    waypoints: Optional[List[Dict[str, float]]] = None


@dataclass
class _Group:
    id: str
    nodes: List[str]
    minX: float = math.inf
    maxX: float = -math.inf
    minY: float = math.inf
    maxY: float = -math.inf
    height: float = 0.0


class Graph:
    def __init__(self) -> None:
        self.nodes: List[_Node] = []
        self.edges: List[_Edge] = []
        self.groups: List[_Group] = []
        self.groupOrder: List[str] = []

    def addNode(self, node_id: str, label: str) -> "Graph":
        self.nodes.append(_Node(id=str(node_id), label=str(label)))
        return self

    def addEdge(self, source: str, target: str) -> "Graph":
        self.edges.append(_Edge(source=str(source), target=str(target)))
        return self

    def getNode(self, node_id: str) -> Optional[_Node]:
        nid = str(node_id)
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def getOutgoingEdges(self, node_id: str) -> List[_Edge]:
        nid = str(node_id)
        return [e for e in self.edges if e.source == nid]

    def getIncomingEdges(self, node_id: str) -> List[_Edge]:
        nid = str(node_id)
        return [e for e in self.edges if e.target == nid]

    def createGroup(self, groupId: str, nodeIds: List[str], *, isHorizontalLayout: bool) -> "Graph":
        gid = str(groupId)
        for node_id in nodeIds:
            node = self.getNode(node_id)
            if node:
                node.group = gid

        self.groups.append(_Group(id=gid, nodes=[str(x) for x in nodeIds]))
        self.groupOrder.append(gid)

        def get_group_avg_pos(gid2: str) -> Tuple[float, float]:
            nodes_in = [n for n in self.nodes if n.group == gid2]
            if not nodes_in:
                return (0.0, 0.0)
            sx = sum((n.x or 0.0) for n in nodes_in)
            sy = sum((n.y or 0.0) for n in nodes_in)
            return (sx / len(nodes_in), sy / len(nodes_in))

        def _sort_key(gid2: str) -> float:
            x, y = get_group_avg_pos(gid2)
            return y if isHorizontalLayout else x

        self.groupOrder.sort(key=_sort_key)
        return self

    def getNodesInGroup(self, groupId: str) -> List[_Node]:
        gid = str(groupId)
        return [n for n in self.nodes if n.group == gid]

    def getGroup(self, groupId: str) -> Optional[_Group]:
        gid = str(groupId)
        for g in self.groups:
            if g.id == gid:
                return g
        return None


class SugiyamaLayout:
    """
    Minimal faithful port of `SugiyamaLayout` used by createAutoLayout().
    Note: In JS, `minimizeCrossings()` is commented out in `run()`.
    """

    def __init__(self, graph: Graph, *, isHorizontalLayout: bool = False) -> None:
        self.graph = graph
        self.layers: List[List[_Node]] = []
        self.isHorizontalLayout = bool(isHorizontalLayout)

    # ---------------------------
    # Step 0: mark feedback edges
    # ---------------------------
    def markFeedbackEdges(self) -> "SugiyamaLayout":
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            stack.add(node_id)
            for edge in self.graph.getOutgoingEdges(node_id):
                target_id = edge.target
                if target_id in stack:
                    edge.feedback = True
                else:
                    dfs(target_id)
            stack.remove(node_id)

        for n in self.graph.nodes:
            dfs(n.id)
        return self

    # ---------------------------
    # Step 1: assign layers (BFS)
    # ---------------------------
    def assignLayers(self) -> "SugiyamaLayout":
        for n in self.graph.nodes:
            n.layer = 0

        assigned: set[str] = set()
        queue: List[str] = []

        # roots = nodes with no incoming edges
        for n in self.graph.nodes:
            if len(self.graph.getIncomingEdges(n.id)) == 0:
                queue.append(n.id)
                assigned.add(n.id)
                n.layer = 0

        while queue:
            current_id = queue.pop(0)
            current_node = self.graph.getNode(current_id)
            if not current_node:
                continue
            for edge in [e for e in self.graph.getOutgoingEdges(current_id) if not e.feedback]:
                target_node = self.graph.getNode(edge.target)
                if not target_node:
                    continue
                target_node.layer = max(int(target_node.layer or 0), int(current_node.layer or 0) + 1)
                if edge.target not in assigned:
                    queue.append(edge.target)
                    assigned.add(edge.target)

        max_layer = 0
        for n in self.graph.nodes:
            max_layer = max(max_layer, int(n.layer or 0))

        self.layers = []
        for i in range(max_layer + 1):
            self.layers.append([n for n in self.graph.nodes if int(n.layer or 0) == i])

        return self

    # ---------------------------
    # Step 3: assign coordinates
    # ---------------------------
    def initializeGroupBoundaries(self) -> None:
        for g in self.graph.groups:
            g.minX = math.inf
            g.maxX = -math.inf
            g.minY = math.inf
            g.maxY = -math.inf

    def calculateGroupHorizontalRanges(self, totalWidth: float, spacing: float) -> Dict[str, Dict[str, float]]:
        ranges: Dict[str, Dict[str, float]] = {}
        groupCount = len(self.graph.groupOrder)
        if groupCount == 0:
            return ranges

        groupLayerDensity: Dict[str, Dict[str, Any]] = {}
        for groupId in self.graph.groupOrder:
            nodes_in = self.graph.getNodesInGroup(groupId)
            layerCounts: Dict[int, int] = {}
            maxNodesInLayer = 0
            for node in nodes_in:
                layer = int(node.layer or 0)
                layerCounts[layer] = layerCounts.get(layer, 0) + 1
                maxNodesInLayer = max(maxNodesInLayer, layerCounts[layer])
            groupLayerDensity[groupId] = {"layerCounts": layerCounts, "maxNodesInLayer": maxNodesInLayer}

        currentX = 0.0
        # JS: Object.keys(groupLayerDensity).forEach(...) (in insertion order)
        for groupId in list(groupLayerDensity.keys()):
            group = groupLayerDensity[groupId]
            baseWidth = 120.0
            groupWidth = baseWidth * float(group.get("maxNodesInLayer") or 1)
            ranges[groupId] = {"minX": currentX, "maxX": currentX + groupWidth}
            currentX += groupWidth

        return ranges

    def ensureNodeWithinLaneBounds(self, node: _Node, laneMinX: float, laneMaxX: float, nodeMargin: float) -> _Node:
        if not node or not node.width:
            return node
        half = float(node.width) / 2.0
        leftBound = laneMinX + nodeMargin + half
        rightBound = laneMaxX - nodeMargin - half
        if node.x < leftBound:
            node.x = leftBound
        if node.x > rightBound:
            node.x = rightBound
        return node

    def updateGroupBoundaries(self, layerHeight: float) -> "SugiyamaLayout":
        globalMaxLayer = 0
        for n in self.graph.nodes:
            globalMaxLayer = max(globalMaxLayer, int(n.layer or 0))

        groupRanges = self.calculateGroupHorizontalRanges(0, 0)

        for groupId in self.graph.groupOrder:
            group = self.graph.getGroup(groupId)
            if not group:
                continue
            gr = groupRanges.get(groupId)
            if not gr:
                continue
            laneMinX = float(gr["minX"])
            laneMaxX = float(gr["maxX"])
            group.minX = laneMinX
            group.maxX = laneMaxX
            totalHeight = float((globalMaxLayer + 1) * layerHeight)
            group.minY = 0.0
            group.maxY = totalHeight
            group.height = group.maxY - group.minY

            nodeMargin = 10.0
            for node in self.graph.nodes:
                if node.group == groupId:
                    self.ensureNodeWithinLaneBounds(node, laneMinX, laneMaxX, nodeMargin)

        return self

    def assignCoordinates(self) -> "SugiyamaLayout":
        defaultNodeWidth = 100.0
        defaultNodeHeight = 40.0
        layerHeight = 150.0
        horizontalSpacing = 20.0
        minNodeMargin = 10.0

        self.initializeGroupBoundaries()
        groupRanges = self.calculateGroupHorizontalRanges(0, 0)

        for i, layer in enumerate(self.layers):
            nodesPerGroup: Dict[Optional[str], List[_Node]] = {}
            for node in layer:
                nodesPerGroup.setdefault(node.group, []).append(node)

            for groupId, groupNodes in nodesPerGroup.items():
                if groupId is None:
                    # JS treats null group keys too; skip if no range.
                    continue
                groupRange = groupRanges.get(groupId)
                if not groupRange:
                    continue

                totalNodesWidth = len(groupNodes) * defaultNodeWidth
                totalSpacing = horizontalSpacing * max(0, len(groupNodes) - 1)
                availableWidth = float(groupRange["maxX"] - groupRange["minX"] - (2 * minNodeMargin))
                actualSpacing = horizontalSpacing
                if len(groupNodes) > 1 and (totalNodesWidth + totalSpacing) > availableWidth:
                    actualSpacing = max(5.0, (availableWidth - totalNodesWidth) / float(len(groupNodes) - 1))

                startX = float(groupRange["minX"] + minNodeMargin)
                currentX = startX

                groupNodes.sort(key=lambda n: int(n.order or 0))
                for node in groupNodes:
                    width = defaultNodeWidth
                    height = float(node.height or defaultNodeHeight)
                    nodeX = currentX + width / 2.0
                    leftBound = float(groupRange["minX"] + minNodeMargin)
                    rightBound = float(groupRange["maxX"] - minNodeMargin)
                    if nodeX - width / 2.0 < leftBound:
                        nodeX = leftBound + width / 2.0
                    if nodeX + width / 2.0 > rightBound:
                        nodeX = rightBound - width / 2.0

                    node.x = nodeX
                    node.y = float(i * layerHeight + 40.0)
                    currentX = nodeX + width / 2.0 + actualSpacing

        self.updateGroupBoundaries(layerHeight)
        return self

    def adjustNodePositionsForGroups(self) -> "SugiyamaLayout":
        for node in self.graph.nodes:
            if not node.group:
                continue
            group = self.graph.getGroup(node.group)
            if not group:
                continue
            self.ensureNodeWithinLaneBounds(node, group.minX, group.maxX, 10.0)
        return self

    # ---------------------------
    # Edge routing (generateEdgeCoordinates)
    # ---------------------------
    def generateEdgeCoordinates(self) -> "SugiyamaLayout":
        spacing = 20.0
        gridSize = 20.0
        maxSteps = 300

        def get_all_obstacles() -> List[Dict[str, Any]]:
            out = []
            for node in self.graph.nodes:
                if not node.id or not node.width or not node.height:
                    continue
                out.append(
                    {
                        "id": node.id,
                        "x": float(node.x),
                        "y": float(node.y),
                        "width": float(node.width) + spacing,
                        "height": float(node.height) + spacing,
                        "type": "node",
                    }
                )
            # group obstacles are not included in JS default getAllObstacles() (it only returns nodes),
            # but getBoundaryPoint expects 'group' type sometimes. We'll omit for fidelity.
            return out

        def intersects_obstacle(p: Dict[str, float], obstacles: List[Dict[str, Any]]) -> bool:
            px, py = float(p["x"]), float(p["y"])
            for obs in obstacles:
                left = float(obs["x"]) - float(obs["width"]) / 2.0
                right = float(obs["x"]) + float(obs["width"]) / 2.0
                top = float(obs["y"]) - float(obs["height"]) / 2.0
                bottom = float(obs["y"]) + float(obs["height"]) / 2.0
                if left <= px <= right and top <= py <= bottom:
                    return True
            return False

        def manhattan(a: Dict[str, float], b: Dict[str, float]) -> float:
            return abs(float(a["x"]) - float(b["x"])) + abs(float(a["y"]) - float(b["y"]))

        def serialize_point(p: Dict[str, float]) -> str:
            return f'{int(round(p["x"]))},{int(round(p["y"]))}'

        def optimize_path(path: List[Dict[str, float]]) -> List[Dict[str, float]]:
            if len(path) <= 2:
                return path
            eps = 0.5

            def are_colinear(p1, p2, p3) -> bool:
                dx1 = float(p2["x"]) - float(p1["x"])
                dy1 = float(p2["y"]) - float(p1["y"])
                dx2 = float(p3["x"]) - float(p2["x"])
                dy2 = float(p3["y"]) - float(p2["y"])
                return abs(dx1 * dy2 - dy1 * dx2) < eps

            result = [path[0]]
            prev = path[0]
            for i in range(1, len(path) - 1):
                curr = path[i]
                nxt = path[i + 1]
                if not are_colinear(prev, curr, nxt):
                    result.append(curr)
                    prev = curr
            result.append(path[-1])
            return result

        def find_orthogonal_path(start: Dict[str, float], end: Dict[str, float], obstacles: List[Dict[str, Any]]) -> List[Dict[str, float]]:
            # BFS-ish with pruning, closely mirroring JS
            start = {"x": float(start["x"]), "y": float(start["y"])}
            end = {"x": float(end["x"]), "y": float(end["y"])}

            queue: List[Dict[str, Any]] = [{"point": start, "path": [start], "turns": 0}]
            visited = {serialize_point(start)}

            def get_direction_priority(point: Dict[str, float], target: Dict[str, float], curr_dir: Optional[Dict[str, float]]) -> List[Dict[str, float]]:
                dx = float(target["x"]) - float(point["x"])
                dy = float(target["y"]) - float(point["y"])
                directions: List[Dict[str, float]] = []
                if abs(dx) > abs(dy):
                    directions.append({"dx": gridSize * (1 if dx > 0 else -1), "dy": 0})
                    directions.append({"dx": 0, "dy": gridSize * (1 if dy > 0 else -1)})
                else:
                    directions.append({"dx": 0, "dy": gridSize * (1 if dy > 0 else -1)})
                    directions.append({"dx": gridSize * (1 if dx > 0 else -1), "dy": 0})

                # add all 4
                directions.extend(
                    [
                        {"dx": gridSize, "dy": 0},
                        {"dx": -gridSize, "dy": 0},
                        {"dx": 0, "dy": gridSize},
                        {"dx": 0, "dy": -gridSize},
                    ]
                )

                # unique
                seen = set()
                uniq = []
                for d in directions:
                    key = f'{int(d["dx"])},{int(d["dy"])}'
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(d)
                return uniq

            bestPath: Optional[List[Dict[str, float]]] = None
            bestTurns = math.inf

            for _ in range(maxSteps):
                if len(queue) > 1000:
                    queue.sort(key=lambda s: s["turns"] * 10 + manhattan(s["point"], end))
                    queue = queue[:100]
                if not queue:
                    break

                queue.sort(key=lambda s: (s["turns"], manhattan(s["point"], end)))
                state = queue.pop(0)
                point = state["point"]
                path = state["path"]
                turns = int(state["turns"])

                if bestPath is not None and turns >= bestTurns:
                    continue

                if manhattan(point, end) < gridSize:
                    fullPath = list(path) + [end]
                    pathTurns = 0
                    for i in range(1, len(fullPath) - 1):
                        prev = fullPath[i - 1]
                        curr = fullPath[i]
                        nxt = fullPath[i + 1]
                        if not ((prev["x"] == curr["x"] == nxt["x"]) or (prev["y"] == curr["y"] == nxt["y"])):
                            pathTurns += 1
                    if pathTurns < bestTurns:
                        bestPath = fullPath
                        bestTurns = pathTurns
                    if pathTurns <= 2:
                        return bestPath
                    continue

                currDirection = None
                if len(path) > 1:
                    prev = path[-2]
                    currDirection = {"dx": float(point["x"]) - float(prev["x"]), "dy": float(point["y"]) - float(prev["y"])}

                for direction in get_direction_priority(point, end, currDirection):
                    nxt = {
                        "x": round((float(point["x"]) + float(direction["dx"])) / gridSize) * gridSize,
                        "y": round((float(point["y"]) + float(direction["dy"])) / gridSize) * gridSize,
                    }
                    key = serialize_point(nxt)
                    if key in visited:
                        continue
                    visited.add(key)

                    if intersects_obstacle(nxt, obstacles):
                        continue
                    if len(path) > maxSteps / 2:
                        continue

                    newTurns = turns
                    if len(path) > 1:
                        prev2 = path[-2]
                        curr2 = point
                        prevDx = float(curr2["x"]) - float(prev2["x"])
                        prevDy = float(curr2["y"]) - float(prev2["y"])
                        if (prevDx != 0 and float(direction["dx"]) == 0) or (prevDy != 0 and float(direction["dy"]) == 0):
                            newTurns += 1

                    queue.append({"point": nxt, "path": list(path) + [nxt], "turns": newTurns})

            if bestPath is not None:
                return optimize_path(bestPath)

            horizontal = abs(end["x"] - start["x"]) > abs(end["y"] - start["y"])
            mid = {"x": end["x"], "y": start["y"]} if horizontal else {"x": start["x"], "y": end["y"]}
            return optimize_path([start, mid, end])

        def get_boundary_point(node: _Node, to: _Node, allObstacles: List[Dict[str, Any]], isStart: bool) -> Dict[str, Any]:
            testLength = 60.0
            directions = {
                "right": {"dx": 1.0, "dy": 0.0},
                "left": {"dx": -1.0, "dy": 0.0},
                "top": {"dx": 0.0, "dy": -1.0},
                "bottom": {"dx": 0.0, "dy": 1.0},
            }

            halfWidth = (float(node.height) / 2.0) if self.isHorizontalLayout else (float(node.width) / 2.0)
            halfHeight = (float(node.width) / 2.0) if self.isHorizontalLayout else (float(node.height) / 2.0)

            portPoints = {
                "left": {"x": float(node.x) - halfWidth, "y": float(node.y), "direction": "left"},
                "right": {"x": float(node.x) + halfWidth, "y": float(node.y), "direction": "right"},
                "top": {"x": float(node.x), "y": float(node.y) - halfHeight, "direction": "top"},
                "bottom": {"x": float(node.x), "y": float(node.y) + halfHeight, "direction": "bottom"},
            }

            priority = {"free": 3, "group": 2, "node": 1}

            def check_direction(dirKey: str) -> str:
                x = float(portPoints[dirKey]["x"])
                y = float(portPoints[dirKey]["y"])
                dx = float(directions[dirKey]["dx"])
                dy = float(directions[dirKey]["dy"])
                status = "free"
                d = 0.0
                while d < testLength:
                    px = x + dx * d
                    py = y + dy * d
                    for obs in allObstacles:
                        left = float(obs["x"]) - float(obs["width"]) / 2.0
                        right = float(obs["x"]) + float(obs["width"]) / 2.0
                        top = float(obs["y"]) - float(obs["height"]) / 2.0
                        bottom = float(obs["y"]) + float(obs["height"]) / 2.0
                        if left <= px <= right and top <= py <= bottom:
                            if obs.get("type") == "node":
                                return "node"
                            if obs.get("type") == "group" and status != "node":
                                status = "group"
                    d += 10.0
                return status

            dx = float(to.x) - float(node.x)
            dy = float(to.y) - float(node.y)
            toDirs: List[str] = []
            if dy < 0:
                toDirs.append("top")
            if dy > 0:
                toDirs.append("bottom")
            if dx > 0:
                toDirs.append("right")
            if dx < 0:
                toDirs.append("left")

            allDirs = ["top", "right", "bottom", "left"]
            ordered = list(toDirs) + [d for d in allDirs if d not in toDirs]

            if isStart:
                ordered = [d for d in ordered if d != "top"]
            else:
                ordered = [d for d in ordered if d != "left"]

            directionScores = {d: check_direction(d) for d in ordered}
            bestDir = sorted(directionScores.items(), key=lambda kv: priority[kv[1]], reverse=True)[0][0]
            return portPoints[bestDir]

        obstacles = get_all_obstacles()
        baseStep = 15.0

        def adjust_initial_step(point: Dict[str, Any]) -> Dict[str, Any]:
            x = float(point["x"])
            y = float(point["y"])
            direction = str(point.get("direction") or "")
            if direction == "left":
                return {"x": x - baseStep, "y": y, "direction": direction}
            if direction == "right":
                return {"x": x + baseStep, "y": y, "direction": direction}
            if direction == "top":
                return {"x": x, "y": y - baseStep, "direction": direction}
            if direction == "bottom":
                return {"x": x, "y": y + baseStep, "direction": direction}
            return {"x": x, "y": y, "direction": direction}

        for edge in self.graph.edges:
            source = self.graph.getNode(edge.source)
            target = self.graph.getNode(edge.target)
            if not source or not target:
                continue

            rawStart = get_boundary_point(source, target, obstacles, True)
            rawEnd = get_boundary_point(target, source, obstacles, False)
            startPoint = adjust_initial_step(rawStart)
            endPoint = adjust_initial_step(rawEnd)
            filtered = [o for o in obstacles if o["id"] not in (source.id, target.id)]
            path = find_orthogonal_path(startPoint, endPoint, filtered)
            edge.waypoints = [{"x": float(rawStart["x"]), "y": float(rawStart["y"])}] + path + [{"x": float(rawEnd["x"]), "y": float(rawEnd["y"])}]

        return self

    def simplifyAllEdgePaths(self) -> "SugiyamaLayout":
        eps = 0.5

        def are_colinear(p1, p2, p3) -> bool:
            dx1 = float(p2["x"]) - float(p1["x"])
            dy1 = float(p2["y"]) - float(p1["y"])
            dx2 = float(p3["x"]) - float(p2["x"])
            dy2 = float(p3["y"]) - float(p2["y"])
            return abs(dx1 * dy2 - dy1 * dx2) < eps

        for edge in self.graph.edges:
            wp = edge.waypoints
            if not wp or len(wp) <= 2:
                continue
            result = [wp[0]]
            prev = wp[0]
            for i in range(1, len(wp) - 1):
                curr = wp[i]
                nxt = wp[i + 1]
                if not are_colinear(prev, curr, nxt):
                    result.append(curr)
                    prev = curr
            result.append(wp[-1])
            edge.waypoints = result
        return self

    def run(self) -> Graph:
        self.markFeedbackEdges().assignLayers().assignCoordinates().calculateGroupHeights().adjustNodePositionsForGroups().generateEdgeCoordinates().simplifyAllEdgePaths()
        return self.graph

    # kept for chaining compatibility
    def calculateGroupHeights(self, layerHeight: float = 120.0) -> "SugiyamaLayout":
        return self


# ---------------------------------------------------------------------------
# BPMN XML generator (ported from BPMNXmlGenerator.vue)
# ---------------------------------------------------------------------------


class ProcessGPTBPMNXmlGenerator:
    def __init__(self) -> None:
        self.PATHS = {
            "TARGET_NAMESPACE": "http://bpmn.io/schema/bpmn",
            "EXPORTER": "Custom BPMN Modeler",
            "EXPORTER_VERSION": "1.0",
        }
        self.DEFAULT_VALUES = {
            "MEGA_PROCESS_ID": "미분류",
            "MAJOR_PROCESS_ID": "미분류",
            "DEFAULT_PROCESS_NAME": "Unknown",
            "DEFAULT_DURATION": 5,
            "FORM_ACTIVITY_TYPE": "org.uengine.kernel.FormActivity",
            "EVALUATE_TYPE": "org.uengine.kernel.Evaluate",
        }

    # ---------------------------
    # helpers ported
    # ---------------------------
    def taskMapping(self, activity_type: str) -> str:
        if activity_type == "ScriptActivity":
            return "bpmn:scriptTask"
        if activity_type == "EmailActivity":
            return "bpmn:sendTask"
        if activity_type == "CallActivity":
            return "bpmn:callActivity"
        if activity_type == "ManualActivity":
            return "bpmn:manualTask"
        return "bpmn:userTask"

    def checkForm(self, variables: Any, variable: str) -> bool:
        if not isinstance(variables, list):
            return False
        form_vars = [d for d in variables if isinstance(d, dict) and d.get("type") == "Form"]
        return any((f.get("name") == variable) for f in form_vars)

    def buildExtension(self, jsonObj: Any) -> ET.Element:
        ext = ET.Element(_q(NS_BPMN, "extensionElements"))
        prop = ET.SubElement(ext, _q(NS_UENGINE, "properties"))
        j = ET.SubElement(prop, _q(NS_UENGINE, "json"))
        j.text = json.dumps(jsonObj or {}, ensure_ascii=False)
        return ext

    # ---------------------------
    # auto layout (createAutoLayout + updateJsonModelWithGraphPositions)
    # ---------------------------
    def getBpmnType(self, elementType: Optional[str]) -> str:
        if not elementType:
            return "bpmn:Task"
        t = str(elementType)
        if t.startswith("bpmn:"):
            return t
        mapping = {
            "StartEvent": "bpmn:StartEvent",
            "EndEvent": "bpmn:EndEvent",
            "UserActivity": "bpmn:UserTask",
            "ServiceActivity": "bpmn:ServiceTask",
            "ScriptActivity": "bpmn:ScriptTask",
            "EmailActivity": "bpmn:SendTask",
            "ManualActivity": "bpmn:ManualTask",
            "ExclusiveGateway": "bpmn:ExclusiveGateway",
            "ParallelGateway": "bpmn:ParallelGateway",
            "Event": "bpmn:IntermediateThrowEvent",
            "Activity": "bpmn:Task",
            "Gateway": "bpmn:ExclusiveGateway",
        }
        return mapping.get(t) or "bpmn:Task"

    def createAutoLayout(self, jsonModel: Dict[str, Any]) -> Dict[str, Any]:
        if not jsonModel or not jsonModel.get("elements"):
            return jsonModel

        elements: List[Dict[str, Any]] = []
        if isinstance(jsonModel.get("elements"), list):
            elements.extend([e for e in jsonModel["elements"] if isinstance(e, dict)])
        else:
            for k, v in (jsonModel.get("elements") or {}).items():
                if isinstance(v, dict):
                    elements.append(v)

        is_horizontal = bool(jsonModel.get("isHorizontal"))
        graph = Graph()

        # add nodes
        for element in [e for e in elements if e.get("elementType") != "Sequence"]:
            graph.addNode(str(element.get("id")), str(element.get("name") or ""))
            node = graph.getNode(str(element.get("id")))
            if not node:
                continue

            # size logic
            if is_horizontal:
                node.width = 80
                node.height = 100
            else:
                node.width = 100
                node.height = 80

            etype = str(element.get("elementType") or "")
            if etype == "Gateway":
                node.width = 50
                node.height = 50
            elif etype == "Activity":
                if is_horizontal:
                    node.width = 80
                    node.height = 100
                else:
                    node.width = 100
                    node.height = 80
            elif etype == "Event":
                node.width = 34
                node.height = 34

            node.nodeType = self.getBpmnType(element.get("type"))

        # add edges
        for element in [e for e in elements if e.get("elementType") == "Sequence"]:
            graph.addEdge(str(element.get("source")), str(element.get("target")))

        # groups (roles)
        roles = jsonModel.get("roles") or []
        if isinstance(roles, list):
            for role in roles:
                if not isinstance(role, dict):
                    continue
                role_name = str(role.get("name") or "")
                role_node_ids = [str(n.get("id")) for n in elements if isinstance(n, dict) and n.get("role") == role_name and n.get("elementType") != "Sequence"]
                role_node_ids = [x for x in role_node_ids if x]
                if role_node_ids:
                    graph.createGroup(role_name, role_node_ids, isHorizontalLayout=is_horizontal)

        layout = SugiyamaLayout(graph, isHorizontalLayout=is_horizontal)
        layout.run()
        return self.updateJsonModelWithGraphPositions(jsonModel, graph)

    def updateJsonModelWithGraphPositions(self, jsonModel: Dict[str, Any], graph: Graph) -> Dict[str, Any]:
        is_array = isinstance(jsonModel.get("elements"), list)
        updated_elements: Any = [] if is_array else {}

        def _swap_if_horizontal(x: float, y: float) -> Tuple[float, float]:
            if bool(jsonModel.get("isHorizontal")):
                return (y, x)
            return (x, y)

        if is_array:
            for element in jsonModel.get("elements") or []:
                if not isinstance(element, dict):
                    continue
                updated = dict(element)
                if element.get("elementType") != "Sequence":
                    node = graph.getNode(str(element.get("id")))
                    if node:
                        x, y = _swap_if_horizontal(float(node.x), float(node.y))
                        updated["x"] = x
                        updated["y"] = y
                        updated["width"] = float(node.width or 100)
                        updated["height"] = float(node.height or 80)
                        updated["layer"] = int(node.layer) if node.layer is not None else None
                        updated["order"] = int(node.order) if node.order is not None else None
                        updated["bpmnType"] = node.nodeType or self.getBpmnType(element.get("type"))
                else:
                    # find corresponding edge
                    for edge in graph.edges:
                        if edge.source == str(element.get("source")) and edge.target == str(element.get("target")):
                            if edge.waypoints:
                                if bool(jsonModel.get("isHorizontal")):
                                    updated["waypoints"] = [{"x": float(pt["y"]), "y": float(pt["x"])} for pt in edge.waypoints]
                                else:
                                    updated["waypoints"] = edge.waypoints
                            break
                updated_elements.append(updated)
        else:
            for key, element in (jsonModel.get("elements") or {}).items():
                if not isinstance(element, dict):
                    continue
                updated = dict(element)
                if element.get("elementType") != "Sequence":
                    node = graph.getNode(str(element.get("id")))
                    if node:
                        x, y = _swap_if_horizontal(float(node.x), float(node.y))
                        updated["x"] = x
                        updated["y"] = y
                        updated["width"] = float(node.width or 100)
                        updated["height"] = float(node.height or 80)
                        updated["layer"] = int(node.layer) if node.layer is not None else None
                        updated["order"] = int(node.order) if node.order is not None else None
                        updated["bpmnType"] = node.nodeType or self.getBpmnType(element.get("type"))
                else:
                    for edge in graph.edges:
                        if edge.source == str(element.get("source")) and edge.target == str(element.get("target")):
                            if edge.waypoints:
                                if bool(jsonModel.get("isHorizontal")):
                                    updated["waypoints"] = [{"x": float(pt["y"]), "y": float(pt["x"])} for pt in edge.waypoints]
                                else:
                                    updated["waypoints"] = edge.waypoints
                            break
                updated_elements[str(key)] = updated

        jsonModel["elements"] = updated_elements

        # update role boundary from group boundaries
        roles = jsonModel.get("roles") or []
        if isinstance(roles, list):
            for group in graph.groups:
                for role in roles:
                    if isinstance(role, dict) and str(role.get("name")) == str(group.id):
                        if bool(jsonModel.get("isHorizontal")):
                            role["boundary"] = {
                                "minX": group.minY,
                                "maxX": group.maxY,
                                "minY": group.minX,
                                "maxY": group.maxX,
                                "width": group.maxY - group.minY,
                                "height": group.maxX - group.minX,
                            }
                        else:
                            role["boundary"] = {
                                "minX": group.minX,
                                "maxX": group.maxX,
                                "minY": group.minY,
                                "maxY": group.maxY,
                                "width": group.maxX - group.minX,
                                "height": group.maxY - group.minY,
                            }
        return jsonModel

    # ---------------------------
    # XML generation helpers (initializeXmlDocument etc)
    # ---------------------------
    def initializeXmlDocument(self, jsonModel: Dict[str, Any]) -> ET.ElementTree:
        root = ET.Element(
            _q(NS_BPMN, "definitions"),
            {
                "id": f'Definitions_{jsonModel.get("processDefinitionId") or "default"}',
                "targetNamespace": self.PATHS["TARGET_NAMESPACE"],
                "exporter": self.PATHS["EXPORTER"],
                "exporterVersion": self.PATHS["EXPORTER_VERSION"],
            },
        )
        # ElementTree will output xmlns based on registered namespaces.
        return ET.ElementTree(root)

    def createCollaborationAndProcess(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any]) -> Tuple[ET.Element, ET.Element]:
        root = xmlDoc.getroot()

        collaboration = ET.SubElement(root, _q(NS_BPMN, "collaboration"), {"id": "Collaboration_1"})
        process = ET.SubElement(
            root,
            _q(NS_BPMN, "process"),
            {
                "id": "Process_1",
                "name": str(jsonModel.get("processDefinitionName") or self.DEFAULT_VALUES["DEFAULT_PROCESS_NAME"]),
                "isExecutable": "true",
                "megaProcessId": str(jsonModel.get("megaProcessId") or self.DEFAULT_VALUES["MEGA_PROCESS_ID"]),
                "majorProcessId": str(jsonModel.get("majorProcessId") or self.DEFAULT_VALUES["MAJOR_PROCESS_ID"]),
            },
        )

        participant = ET.SubElement(
            collaboration,
            _q(NS_BPMN, "participant"),
            {
                "id": "Participant",
                "name": str(jsonModel.get("processDefinitionName") or self.DEFAULT_VALUES["DEFAULT_PROCESS_NAME"]),
                "processRef": "Process_1",
            },
        )
        _ = participant
        return collaboration, process

    def createDataElements(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], process_el: ET.Element) -> None:
        data = jsonModel.get("data") or []
        if not isinstance(data, list) or not data:
            return
        ext = ET.SubElement(process_el, _q(NS_BPMN, "extensionElements"))
        props = ET.SubElement(ext, _q(NS_UENGINE, "properties"))
        for dataObj in data:
            if not isinstance(dataObj, dict):
                continue
            if not dataObj.get("name"):
                continue
            var = ET.SubElement(props, _q(NS_UENGINE, "variable"), {"name": str(dataObj.get("name"))})
            typ = str((dataObj.get("type") or "string")).lower()
            var.set("type", typ)
            j = ET.SubElement(var, _q(NS_UENGINE, "json"))
            jsonData: Dict[str, Any] = {"value": dataObj.get("defaultValue") or "", "options": {}}
            if dataObj.get("description"):
                jsonData["options"]["description"] = dataObj.get("description")
            j.text = json.dumps(jsonData, ensure_ascii=False)

    def createSequenceFlows(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], process_el: ET.Element) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
        inComing: Dict[str, List[str]] = {}
        outGoing: Dict[str, List[str]] = {}
        positionMapping: Dict[str, Dict[str, str]] = {}

        sequences: List[Dict[str, Any]] = []
        elems = jsonModel.get("elements")
        if isinstance(elems, list):
            sequences = [e for e in elems if isinstance(e, dict) and e.get("elementType") == "Sequence"]
        elif isinstance(elems, dict):
            for _, e in elems.items():
                if isinstance(e, dict) and e.get("elementType") == "Sequence":
                    sequences.append(e)

        for sequence in sequences:
            if not sequence.get("source") or not sequence.get("target"):
                continue
            seq_id = f'SequenceFlow_{sequence.get("source")}_{sequence.get("target")}'
            sf = ET.SubElement(
                process_el,
                _q(NS_BPMN, "sequenceFlow"),
                {"id": seq_id, "sourceRef": str(sequence.get("source")), "targetRef": str(sequence.get("target"))},
            )

            if sequence.get("condition"):
                ext = ET.SubElement(sf, _q(NS_BPMN, "extensionElements"))
                prop = ET.SubElement(ext, _q(NS_UENGINE, "properties"))
                j = ET.SubElement(prop, _q(NS_UENGINE, "json"))
                cond = sequence.get("condition")
                conditionPayload: Dict[str, Any] = {"condition": None}
                if isinstance(cond, str):
                    conditionPayload["condition"] = cond
                    sf.set("name", cond or "")
                elif isinstance(cond, dict):
                    conditionPayload = {
                        "condition": {
                            "_type": cond.get("_type") or "org.uengine.kernel.Evaluate",
                            "key": cond.get("key") or "",
                            "value": cond.get("value") or "",
                            "condition": cond.get("condition") or "==",
                        }
                    }
                j.text = json.dumps(conditionPayload, ensure_ascii=False)

            inComing.setdefault(str(sequence.get("target")), []).append(seq_id)
            outGoing.setdefault(str(sequence.get("source")), []).append(seq_id)
            positionMapping[seq_id] = {"source": str(sequence.get("source")), "target": str(sequence.get("target"))}

        return inComing, outGoing, positionMapping

    # ---------------------------
    # process element creators
    # ---------------------------
    def createActivity(self, element: Dict[str, Any], data: List[Dict[str, Any]], laneMap: Dict[str, List[str]], outMap: Dict[str, List[str]], inMap: Dict[str, List[str]], process_el: ET.Element) -> None:
        role = str(element.get("role") or "")
        laneMap.setdefault(role, []).append(str(element.get("id")))

        taskType = self.taskMapping(str(element.get("type") or ""))
        local = taskType.split(":", 1)[1] if ":" in taskType else taskType
        task = ET.SubElement(process_el, _q(NS_BPMN, local))
        task.set("id", str(element.get("id")))
        task.set("name", str(element.get("name") or ""))

        for inc_id in inMap.get(str(element.get("id")), []) or []:
            inc = ET.SubElement(task, _q(NS_BPMN, "incoming"))
            inc.text = str(inc_id)
        for out_id in outMap.get(str(element.get("id")), []) or []:
            out = ET.SubElement(task, _q(NS_BPMN, "outgoing"))
            out.text = str(out_id)

        paramObj: Dict[str, Any] = {}
        if element.get("description"):
            paramObj["description"] = element.get("description")
        if element.get("role"):
            paramObj["role"] = element.get("role")

        def to_mapping_obj(lst: List[str], direction_key: str) -> List[Dict[str, Any]]:
            outl: List[Dict[str, Any]] = []
            for item in lst:
                outl.append(
                    {
                        "dataFieldId": item,
                        direction_key: "변수명",
                        "type": self.DEFAULT_VALUES["FORM_ACTIVITY_TYPE"] if self.checkForm(data, item) else self.DEFAULT_VALUES["EVALUATE_TYPE"],
                    }
                )
            return outl

        if isinstance(element.get("inputData"), list) and element.get("inputData"):
            paramObj["inputMapping"] = to_mapping_obj(element.get("inputData"), "to")
        if isinstance(element.get("outputData"), list) and element.get("outputData"):
            paramObj["outputMapping"] = to_mapping_obj(element.get("outputData"), "from")
        if isinstance(element.get("checkpoints"), list) and element.get("checkpoints"):
            paramObj["checkpoints"] = element.get("checkpoints")
        if isinstance(element.get("properties"), dict):
            paramObj.update(element.get("properties"))

        if "duration" in paramObj:
            try:
                paramObj["duration"] = float(paramObj["duration"])
            except Exception:
                pass

        if paramObj:
            task.append(self.buildExtension(paramObj))

    def createCallActivity(self, element: Dict[str, Any], data: List[Dict[str, Any]], laneMap: Dict[str, List[str]], outMap: Dict[str, List[str]], inMap: Dict[str, List[str]], process_el: ET.Element) -> None:
        role = str(element.get("role") or "")
        laneMap.setdefault(role, []).append(str(element.get("id")))
        taskType = self.taskMapping(str(element.get("type") or "CallActivity"))
        local = taskType.split(":", 1)[1] if ":" in taskType else taskType
        task = ET.SubElement(process_el, _q(NS_BPMN, local))
        task.set("id", str(element.get("id")))
        task.set("name", str(element.get("name") or ""))

    def createGateway(self, element: Dict[str, Any], laneMap: Dict[str, List[str]], outMap: Dict[str, List[str]], inMap: Dict[str, List[str]], process_el: ET.Element) -> None:
        role = str(element.get("role") or "")
        laneMap.setdefault(role, []).append(str(element.get("id")))
        gwType = str(element.get("gateWayType") or "bpmn:exclusiveGateway")
        local = gwType.split(":", 1)[1] if ":" in gwType else gwType
        gw = ET.SubElement(process_el, _q(NS_BPMN, local))
        gw.set("id", str(element.get("id")))
        gw.set("name", str(element.get("name") or ""))

        for inc_id in inMap.get(str(element.get("id")), []) or []:
            inc = ET.SubElement(gw, _q(NS_BPMN, "incoming"))
            inc.text = str(inc_id)
        for out_id in outMap.get(str(element.get("id")), []) or []:
            out = ET.SubElement(gw, _q(NS_BPMN, "outgoing"))
            out.text = str(out_id)

        if element.get("description"):
            gw.append(self.buildExtension({"description": element.get("description")}))

    def attachEventDefinition(self, evt: ET.Element, element: Dict[str, Any]) -> None:
        et = element.get("eventType")
        if et == "Timer":
            ET.SubElement(evt, _q(NS_BPMN, "timerEventDefinition"))
        elif et == "Message":
            ET.SubElement(evt, _q(NS_BPMN, "messageEventDefinition"))

    def createEvent(self, element: Dict[str, Any], process_el: ET.Element) -> ET.Element:
        if element.get("type") == "StartEvent":
            evt_local = "startEvent"
        elif element.get("type") == "EndEvent":
            evt_local = "endEvent"
        else:
            evt_local = "intermediateThrowEvent"

        evt = ET.SubElement(process_el, _q(NS_BPMN, evt_local))
        evt.set("id", str(element.get("id")))
        evt.set("name", str(element.get("name") or ""))
        self.attachEventDefinition(evt, element)

        props: Dict[str, Any] = {}
        if element.get("description"):
            props["description"] = element.get("description")
        if element.get("expression"):
            props["expression"] = element.get("expression")
        if props:
            evt.append(self.buildExtension(props))
        return evt

    def createProcessElements(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], process_el: ET.Element, inComing: Dict[str, List[str]], outGoing: Dict[str, List[str]]) -> Dict[str, List[str]]:
        laneActivityMapping: Dict[str, List[str]] = {}

        displayable: List[Dict[str, Any]] = []
        elems = jsonModel.get("elements")
        if isinstance(elems, list):
            for e in elems:
                if isinstance(e, dict) and e.get("elementType") and e.get("elementType") != "Sequence":
                    displayable.append(e)
        elif isinstance(elems, dict):
            for _, e in elems.items():
                if isinstance(e, dict) and e.get("elementType") and e.get("elementType") != "Sequence":
                    displayable.append(e)

        for element in displayable:
            etype = element.get("elementType")
            if etype == "Activity":
                self.createActivity(element, jsonModel.get("data") or [], laneActivityMapping, outGoing, inComing, process_el)
            elif etype == "CallActivity":
                self.createCallActivity(element, jsonModel.get("data") or [], laneActivityMapping, outGoing, inComing, process_el)
            elif etype == "Gateway":
                self.createGateway(element, laneActivityMapping, outGoing, inComing, process_el)
            elif etype == "Event":
                role = str(element.get("role") or "")
                laneActivityMapping.setdefault(role, []).append(str(element.get("id")))
                evt = self.createEvent(element, process_el)
                for seqId in inComing.get(str(element.get("id")), []) or []:
                    inc = ET.SubElement(evt, _q(NS_BPMN, "incoming"))
                    inc.text = str(seqId)
                for seqId in outGoing.get(str(element.get("id")), []) or []:
                    out = ET.SubElement(evt, _q(NS_BPMN, "outgoing"))
                    out.text = str(seqId)

        # subProcesses handling is intentionally omitted here; createBpmnXml in the original supports it,
        # but typical ProcessGPT generation path does not use it. It can be added later if needed.
        return laneActivityMapping

    def createLaneSet(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], process_el: ET.Element, laneActivityMapping: Dict[str, List[str]]) -> Optional[ET.Element]:
        roles = jsonModel.get("roles")
        if not isinstance(roles, list):
            return None
        laneSet = ET.SubElement(process_el, _q(NS_BPMN, "laneSet"), {"id": "LaneSet_1"})
        for idx, role in enumerate(roles):
            if not isinstance(role, dict):
                continue
            lane = ET.SubElement(laneSet, _q(NS_BPMN, "lane"), {"id": f"Lane_{idx}", "name": str(role.get("name") or "")})

            if role.get("endpoint") is not None and role.get("endpoint") != "":
                endpointPayload: Dict[str, Any] = {"roleResolutionContext": {"endpoint": role.get("endpoint")}}
                if role.get("endpoint") == "external_customer":
                    endpointPayload["roleResolutionContext"]["_type"] = "org.uengine.kernel.ExternalCustomerRoleResolutionContext"
                lane.append(self.buildExtension(endpointPayload))

            if role.get("resolutionRule") is not None:
                lane.set("resolutionRule", str(role.get("resolutionRule")))

            for act_id in laneActivityMapping.get(str(role.get("name") or ""), []) or []:
                fn = ET.SubElement(lane, _q(NS_BPMN, "flowNodeRef"))
                fn.text = str(act_id)
        return laneSet

    def createDiagram(self, xmlDoc: ET.ElementTree) -> Tuple[ET.Element, ET.Element]:
        root = xmlDoc.getroot()
        diagram = ET.SubElement(root, _q(NS_BPMNDI, "BPMNDiagram"), {"id": "BPMNDiagram_1"})
        plane = ET.SubElement(diagram, _q(NS_BPMNDI, "BPMNPlane"), {"id": "BPMNPlane_1", "bpmnElement": "Collaboration_1"})
        return diagram, plane

    # ---------------------------
    # DI shapes / edges
    # ---------------------------
    def createElementShape(self, element: Dict[str, Any], elementX: float, elementY: float, isHorizontal: bool, currentSource: str) -> Tuple[ET.Element, Dict[str, Any], Dict[str, Any]]:
        shape = ET.Element(_q(NS_BPMNDI, "BPMNShape"))
        if element.get("elementType") == "Event":
            shape.set("id", f"Shape_{element.get('id')}")
        else:
            shape.set("id", f"BPMNShape_{element.get('id')}")
        shape.set("bpmnElement", str(element.get("id")))
        if element.get("elementType") == "Gateway":
            shape.set("isMarkerVisible", "true")

        bounds = ET.SubElement(shape, _q(NS_DC, "Bounds"))
        width = 50.0
        height = 50.0
        if element.get("elementType") in ("Activity", "CallActivity"):
            width = 100.0
            height = 80.0
        elif element.get("elementType") == "Event":
            width = 34.0
            height = 34.0

        bounds.set("width", str(int(width)))
        bounds.set("height", str(int(height)))
        topLeftX = float(elementX) - width / 2.0
        topLeftY = float(elementY) - height / 2.0
        bounds.set("x", str(topLeftX))
        bounds.set("y", str(topLeftY))

        if element.get("name"):
            label = ET.SubElement(shape, _q(NS_BPMNDI, "BPMNLabel"))
            lb = ET.SubElement(label, _q(NS_DC, "Bounds"))
            lb.set("x", str(topLeftX))
            lb.set("y", str(topLeftY + height + 5))
            lb.set("width", str(int(width)))
            lb.set("height", "14")

        activityPosInfo = {str(element.get("id")): {"x": round(elementX), "y": round(elementY), "width": width, "height": height}}
        offsetPosInfo: Dict[str, Any] = {str(element.get("id")): {}}
        if ((element.get("source") and currentSource == element.get("source")) or element.get("source") == ""):
            offsetPosInfo[str(element.get("id"))] = {
                "topLeftX": topLeftX,
                "topLeftY": topLeftY,
                "center": {"x": elementX, "y": elementY},
                "topLeft": {"x": topLeftX, "y": topLeftY},
                "topRight": {"x": topLeftX + width, "y": topLeftY},
                "bottomLeft": {"x": topLeftX, "y": topLeftY + height},
                "bottomRight": {"x": topLeftX + width, "y": topLeftY + height},
                "top": {"x": topLeftX + width / 2.0, "y": topLeftY},
                "right": {"x": topLeftX + width, "y": topLeftY + height / 2.0},
                "bottom": {"x": topLeftX + width / 2.0, "y": topLeftY + height},
                "left": {"x": topLeftX, "y": topLeftY + height / 2.0},
            }
        return shape, activityPosInfo, offsetPosInfo

    def createShapes(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], plane: ET.Element, isHorizontal: bool) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        activityPos: Dict[str, Any] = {}
        offsetPos: Dict[str, Any] = {}
        roleVector: Dict[str, Any] = {}

        elems = jsonModel.get("elements")
        displayable: List[Dict[str, Any]] = []
        if isinstance(elems, list):
            displayable = [e for e in elems if isinstance(e, dict) and e.get("elementType") and e.get("elementType") != "Sequence"]
        elif isinstance(elems, dict):
            for _, e in elems.items():
                if isinstance(e, dict) and e.get("elementType") and e.get("elementType") != "Sequence":
                    displayable.append(e)

        currentSource = "default"
        roles = jsonModel.get("roles") or []

        for element in displayable:
            elementX = 100.0
            elementY = 100.0
            # JS: if(element.x && element.y) { ... }
            if element.get("x") and element.get("y"):
                elementX = float(element.get("x"))
                elementY = float(element.get("y"))

                elementWidth = float(element.get("width") or (50 if element.get("elementType") == "Gateway" else 34 if element.get("elementType") == "Event" else 100))
                # elementHeight used only for boundary checks
                _ = elementWidth

                if element.get("role") and isinstance(roles, list):
                    for role in roles:
                        if not isinstance(role, dict):
                            continue
                        if role.get("name") != element.get("role"):
                            continue
                        boundary = role.get("boundary")
                        if isinstance(boundary, dict):
                            minX = float(boundary.get("minX", 0))
                            maxX = float(boundary.get("maxX", 0))
                            # JS uses elementWidth/2 and +10 padding
                            if elementX < minX + (elementWidth / 2.0) + 10:
                                elementX = minX + (elementWidth / 2.0) + 10
                            elif elementX > maxX - (elementWidth / 2.0) - 10:
                                elementX = maxX - (elementWidth / 2.0) - 10

                shape, ap, op = self.createElementShape(element, elementX, elementY, isHorizontal, currentSource)
                plane.append(shape)
                activityPos.update(ap)
                offsetPos.update(op)

                role_name = str(element.get("role") or "")
                roleVector.setdefault(role_name, {})
                roleVector[role_name][str(element.get("id"))] = {"x": elementX, "y": elementY}
            else:
                # fallback branch (non-auto-layout) is intentionally omitted
                shape, ap, op = self.createElementShape(element, elementX, elementY, isHorizontal, currentSource)
                plane.append(shape)
                activityPos.update(ap)
                offsetPos.update(op)
                role_name = str(element.get("role") or "")
                roleVector.setdefault(role_name, {})
                roleVector[role_name][str(element.get("id"))] = {"x": elementX, "y": elementY}

        return activityPos, offsetPos, roleVector

    def createParticipantShapeInAutoLayout(self, xmlDoc: ET.ElementTree, plane: ET.Element, isHorizontal: bool, jsonModel: Dict[str, Any]) -> None:
        participantShape = ET.SubElement(plane, _q(NS_BPMNDI, "BPMNShape"), {"id": "Participant_1", "bpmnElement": "Participant", "isHorizontal": _bool_str(isHorizontal)})
        bounds = ET.SubElement(participantShape, _q(NS_DC, "Bounds"))

        roles = jsonModel.get("roles")
        if isinstance(roles, list):
            roles_with_boundary = [r for r in roles if isinstance(r, dict) and isinstance(r.get("boundary"), dict)]
            if roles_with_boundary:
                minX = min(float(r["boundary"]["minX"]) for r in roles_with_boundary)
                minY = min(float(r["boundary"]["minY"]) for r in roles_with_boundary)
                maxX = max(float(r["boundary"]["maxX"]) for r in roles_with_boundary)
                maxY = max(float(r["boundary"]["maxY"]) for r in roles_with_boundary)
                paddingX = 30.0 if isHorizontal else 0.0
                paddingY = 0.0 if isHorizontal else 30.0
                bounds.set("x", str(minX - paddingX))
                bounds.set("y", str(minY - paddingY))
                bounds.set("width", str(maxX - minX + paddingX))
                bounds.set("height", str(maxY - minY + paddingY))
                return

        # fallback
        bounds.set("x", "0")
        bounds.set("y", "0")
        bounds.set("width", "800")
        bounds.set("height", "600")

    def createLaneShapesInAutoLayout(self, xmlDoc: ET.ElementTree, plane: ET.Element, jsonModel: Dict[str, Any], isHorizontal: bool) -> None:
        roles = jsonModel.get("roles")
        if not isinstance(roles, list):
            return
        for index, role in enumerate(roles):
            if not isinstance(role, dict):
                continue
            boundary = role.get("boundary")
            if not isinstance(boundary, dict):
                continue
            laneShape = ET.SubElement(plane, _q(NS_BPMNDI, "BPMNShape"), {"id": f"BPMNShape_{index}", "bpmnElement": f"Lane_{index}", "isHorizontal": _bool_str(isHorizontal)})
            b = ET.SubElement(laneShape, _q(NS_DC, "Bounds"))
            b.set("x", str(boundary.get("minX")))
            b.set("y", str(boundary.get("minY")))
            b.set("width", str(boundary.get("width")))
            b.set("height", str(boundary.get("height")))
            label = ET.SubElement(laneShape, _q(NS_BPMNDI, "BPMNLabel"))
            lb = ET.SubElement(label, _q(NS_DC, "Bounds"))
            lb.set("x", str(float(boundary.get("minX", 0)) + 10))
            lb.set("y", str(float(boundary.get("minY", 0)) + 5))
            lb.set("width", "100")
            lb.set("height", "14")

    def determineDirection(self, sourceActivityPos: Dict[str, Any], targetActivityPos: Dict[str, Any], isHorizontal: bool, typ: str) -> str:
        if isHorizontal:
            if sourceActivityPos.get("y") == targetActivityPos.get("y"):
                return "right" if typ == "source" else "left"
            if float(targetActivityPos.get("y", 0)) > float(sourceActivityPos.get("y", 0)):
                return "bottom" if typ == "source" else "right"
            return "right" if typ == "source" else "top"
        else:
            if sourceActivityPos.get("x") == targetActivityPos.get("x"):
                return "bottom" if typ == "source" else "top"
            if float(targetActivityPos.get("x", 0)) > float(sourceActivityPos.get("x", 0)):
                return "right" if typ == "source" else "top"
            return "left" if typ == "source" else "top"

    def getPosition(self, pos: Dict[str, Any], direction: str) -> Dict[str, Any]:
        position = {"x": float(pos.get("x", 0)), "y": float(pos.get("y", 0)), "direction": direction}
        if direction == "left":
            position["x"] = float(pos.get("x", 0)) - float(pos.get("width", 0)) / 2.0
        elif direction == "right":
            position["x"] = float(pos.get("x", 0)) + float(pos.get("width", 0)) / 2.0
        elif direction == "top":
            position["y"] = float(pos.get("y", 0)) - float(pos.get("height", 0)) / 2.0
        elif direction == "bottom":
            position["y"] = float(pos.get("y", 0)) + float(pos.get("height", 0)) / 2.0
        return position

    def createWaypoint(self, x: float, y: float) -> ET.Element:
        wp = ET.Element(_q(NS_DI, "waypoint"))
        wp.set("x", str(x))
        wp.set("y", str(y))
        return wp

    def createEdgeWaypoints(self, bpmnEdge: ET.Element, start: Dict[str, Any], end: Dict[str, Any], isHorizontal: bool) -> None:
        bpmnEdge.append(self.createWaypoint(start["x"], start["y"]))
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        if isHorizontal:
            if start["y"] == end["y"]:
                bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
            elif start["direction"] == "right" and end["direction"] == "left":
                midY = float(start["y"]) + dy / 2.0
                bpmnEdge.append(self.createWaypoint(start["x"], midY))
                bpmnEdge.append(self.createWaypoint(end["x"], midY))
                bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
                bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
            else:
                bpmnEdge.append(self.createWaypoint(start["x"], start["y"]))
                if dy >= 0:
                    bpmnEdge.append(self.createWaypoint(start["x"], end["y"]))
                    bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
                else:
                    bpmnEdge.append(self.createWaypoint(end["x"], start["y"]))
                    bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
        else:
            if start["x"] == end["x"]:
                bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
            elif start["direction"] == "bottom" and end["direction"] == "top":
                midX = float(start["x"]) + dx / 2.0
                bpmnEdge.append(self.createWaypoint(midX, start["y"]))
                bpmnEdge.append(self.createWaypoint(midX, end["y"]))
                bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
            else:
                bpmnEdge.append(self.createWaypoint(start["x"], start["y"]))
                if dx >= 0:
                    bpmnEdge.append(self.createWaypoint(start["x"], start["y"]))
                    bpmnEdge.append(self.createWaypoint(end["x"], start["y"]))
                    bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))
                else:
                    bpmnEdge.append(self.createWaypoint(start["x"], end["y"]))
                    bpmnEdge.append(self.createWaypoint(end["x"], end["y"]))

    def createSequenceEdges(self, xmlDoc: ET.ElementTree, jsonModel: Dict[str, Any], plane: ET.Element, offsetPos: Dict[str, Any], activityPos: Dict[str, Any], isHorizontal: bool) -> None:
        sequences: List[Dict[str, Any]] = []
        elems = jsonModel.get("elements")
        if isinstance(elems, list):
            sequences = [e for e in elems if isinstance(e, dict) and e.get("elementType") == "Sequence"]
        elif isinstance(elems, dict):
            for _, e in elems.items():
                if isinstance(e, dict) and e.get("elementType") == "Sequence":
                    sequences.append(e)

        for sequence in sequences:
            if ((not offsetPos.get(str(sequence.get("source"))) or not offsetPos.get(str(sequence.get("target")))) and not sequence.get("waypoints")):
                continue
            edge_el = ET.SubElement(
                plane,
                _q(NS_BPMNDI, "BPMNEdge"),
                {"id": f'BPMNEdge_{sequence.get("source")}_{sequence.get("target")}', "bpmnElement": f'SequenceFlow_{sequence.get("source")}_{sequence.get("target")}'},
            )
            if sequence.get("waypoints"):
                for wp in sequence.get("waypoints") or []:
                    if not isinstance(wp, dict):
                        continue
                    edge_el.append(self.createWaypoint(float(wp.get("x", 0)), float(wp.get("y", 0))))
            else:
                sourcePos = offsetPos.get(str(sequence.get("source"))) or {}
                targetPos = offsetPos.get(str(sequence.get("target"))) or {}
                sourceActivityPos = activityPos.get(str(sequence.get("source"))) or {}
                targetActivityPos = activityPos.get(str(sequence.get("target"))) or {}
                sd = self.determineDirection(sourceActivityPos, targetActivityPos, isHorizontal, "source")
                td = self.determineDirection(sourceActivityPos, targetActivityPos, isHorizontal, "target")
                start = self.getPosition(sourcePos, sd)
                end = self.getPosition(targetPos, td)
                self.createEdgeWaypoints(edge_el, start, end, isHorizontal)

    # ---------------------------
    # public API
    # ---------------------------
    def create_bpmn_xml(self, jsonModel: Dict[str, Any], horizontal: Optional[bool] = None) -> str:
        """
        Backend equivalent of BPMNXmlGenerator.vue#createBpmnXml(jsonModel, horizontal)
        """
        jsonModel = dict(jsonModel or {})
        jsonModel["isAutoLayout"] = True

        isHorizontal = bool(jsonModel.get("isHorizontal"))
        if horizontal is not None:
            isHorizontal = bool(horizontal)
            jsonModel["isHorizontal"] = bool(horizontal)

        jsonModel = self.createAutoLayout(jsonModel)

        # initialize document and build structure
        xmlDoc = self.initializeXmlDocument(jsonModel)
        collaboration, process_el = self.createCollaborationAndProcess(xmlDoc, jsonModel)
        _ = collaboration
        self.createDataElements(xmlDoc, jsonModel, process_el)
        inComing, outGoing, _pos = self.createSequenceFlows(xmlDoc, jsonModel, process_el)
        laneActivityMapping = self.createProcessElements(xmlDoc, jsonModel, process_el, inComing, outGoing)
        self.createLaneSet(xmlDoc, jsonModel, process_el, laneActivityMapping)
        _diagram, plane = self.createDiagram(xmlDoc)

        activityPos, offsetPos, roleVector = self.createShapes(xmlDoc, jsonModel, plane, isHorizontal)
        _ = roleVector

        # participant + lanes in auto layout
        self.createParticipantShapeInAutoLayout(xmlDoc, plane, isHorizontal, jsonModel)
        self.createLaneShapesInAutoLayout(xmlDoc, plane, jsonModel, isHorizontal)

        self.createSequenceEdges(xmlDoc, jsonModel, plane, offsetPos, activityPos, isHorizontal)

        xml_bytes = ET.tostring(xmlDoc.getroot(), encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")

