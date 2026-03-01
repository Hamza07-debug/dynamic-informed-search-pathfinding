"""
╔══════════════════════════════════════════════════════════════════════════╗
║            PathViz — Dynamic Pathfinding Agent                          ║
║            Artificial Intelligence — Informed Search Algorithms         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Algorithms  :  A* Search  (f = g + h)                                  ║
║                 Greedy Best-First Search  (f = h)                        ║
║  Heuristics  :  Manhattan Distance  |  Euclidean Distance                ║
║  Features    :  Dynamic obstacles · Live re-planning · Metrics dashboard ║
╠══════════════════════════════════════════════════════════════════════════╣
║  KEYBOARD                                                                ║
║    Space          Start / restart search                                 ║
║    R              Reset search (keep walls)                              ║
║    C              Clear all obstacles                                    ║
║    [ / ]          Decrease / increase obstacle density by 5 %            ║
║    Escape         Quit                                                   ║
║  MOUSE                                                                   ║
║    Left-click          Toggle wall                                       ║
║    Shift + LMB         Move Start node                                   ║
║    Ctrl  + LMB         Move Goal  node                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard-library imports
# ─────────────────────────────────────────────────────────────────────────────
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party import
# ─────────────────────────────────────────────────────────────────────────────
import pygame


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── CONFIGURATION & COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════════════════

# ── Window ───────────────────────────────────────────────────────────────────
WINDOW_W: int = 1366
WINDOW_H: int = 720
PANEL_W:  int = 340          # right-side panel width — wider for readability
GRID_PAD: int = 8            # padding around grid area

GRID_RECT: pygame.Rect = pygame.Rect(
    GRID_PAD,
    GRID_PAD,
    WINDOW_W - PANEL_W - GRID_PAD * 2,
    WINDOW_H - GRID_PAD * 2,
)

# ── Base palette ─────────────────────────────────────────────────────────────
COL_BG         = (11,  12,  16)    # main canvas background
COL_PANEL_BG   = (18,  20,  28)    # right panel background — slightly lighter
COL_CARD_BG    = (26,  29,  40)    # card / metric box fill
COL_CARD_BG2   = (32,  36,  50)    # secondary card shade (bar bg)
COL_BORDER     = (44,  50,  68)    # default border colour
COL_BORDER_HI  = (64,  72,  98)    # highlighted border (grid glow)

# ── Accent colours ───────────────────────────────────────────────────────────
COL_BLUE       = (60,  165, 255)   # primary accent / info
COL_GREEN      = (44,  210, 110)   # success / path found
COL_AMBER      = (255, 190,  45)   # warning / frontier / density
COL_RED        = (255,  78,  78)   # error / no-path / cost-inf

# ── Text hierarchy ────────────────────────────────────────────────────────────
COL_TEXT1      = (225, 232, 248)   # bright  – primary values
COL_TEXT2      = (155, 166, 192)   # dim     – labels  (was too dark, now legible)
COL_TEXT3      = ( 80,  90, 115)   # faint   – decorative / fps
COL_MUTED      = (110, 120, 155)   # section headings

# ── Grid cell colours  (directly satisfy §5 colour requirements) ─────────────
COL_CELL_EMPTY    = ( 17,  19,  26)
COL_CELL_WALL     = (  7,   7,  10)
COL_CELL_GRID_LN  = ( 26,  28,  38)
COL_CELL_START    = ( 28, 138, 252)   # bright blue  – distinct start
COL_CELL_GOAL     = (138,  62, 238)   # violet       – distinct goal
COL_CELL_FRONTIER = (252, 182,  38)   # ← YELLOW     (requirement §5)
COL_CELL_VISITED  = ( 24,  68, 130)   # ← BLUE       (requirement §5)
COL_CELL_PATH     = ( 36, 196,  98)   # ← GREEN      (requirement §5)
COL_CELL_AGENT    = (  0, 226, 200)   # cyan agent marker

# ── Button colours ────────────────────────────────────────────────────────────
COL_BTN_BG    = (24, 27, 38)
COL_BTN_HOVER = (32, 36, 52)

# ── Simulation parameters ─────────────────────────────────────────────────────
DEFAULT_DENSITY:   float = 0.28    # initial random obstacle fill ratio
MIN_DENSITY:       float = 0.05
MAX_DENSITY:       float = 0.60
STEPS_PER_FRAME:   int   = 5       # node expansions per render frame
AGENT_MOVE_DELAY:  float = 0.028   # seconds between agent path steps
DYN_OBSTACLE_PROB: float = 0.018   # per-frame spawn chance in dynamic mode


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── ENUMERATIONS
# ═════════════════════════════════════════════════════════════════════════════

class AlgorithmType(Enum):
    """Supported informed-search algorithms."""
    ASTAR = "A*"
    GBFS  = "Greedy BFS"


class HeuristicType(Enum):
    """Selectable heuristic distance functions."""
    MANHATTAN = "Manhattan"
    EUCLIDEAN = "Euclidean"


class SearchStatus(Enum):
    """Internal state of a running SearchAlgorithm instance."""
    IDLE    = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


class AgentState(Enum):
    """High-level lifecycle state of the Agent."""
    IDLE      = auto()   # waiting for user to press Start
    SEARCHING = auto()   # algorithm is expanding frontier nodes
    MOVING    = auto()   # agent is traversing the found path
    NO_PATH   = auto()   # search exhausted — no solution exists


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── NODE  (grid cell data model)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """
    One cell in the 2-D grid.

    Holds both *map geometry* (wall / start / goal flags) and
    *search state* (g / h / f costs, open-set membership, parent pointer).
    Separating these into a single dataclass keeps access O(1) and avoids
    parallel arrays.
    """
    row: int
    col: int

    # Map geometry
    is_wall:  bool = False
    is_start: bool = False
    is_goal:  bool = False

    # Search costs (reset before every new search)
    g_cost: float = field(default=float("inf"))
    h_cost: float = field(default=0.0)
    f_cost: float = field(default=float("inf"))

    # Tree / open-set membership (excluded from equality so nodes compare by position)
    parent:        Optional["Node"] = field(default=None,  compare=False)
    in_open_set:   bool             = field(default=False, compare=False)
    in_closed_set: bool             = field(default=False, compare=False)
    in_path:       bool             = field(default=False, compare=False)

    def reset_search_state(self) -> None:
        """Clear all search fields without touching map geometry."""
        self.g_cost        = float("inf")
        self.h_cost        = 0.0
        self.f_cost        = float("inf")
        self.parent        = None
        self.in_open_set   = False
        self.in_closed_set = False
        self.in_path       = False


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── GRID  (environment manager)
# ═════════════════════════════════════════════════════════════════════════════

class Grid:
    """
    Owns and manages the 2-D array of Node objects.

    Responsibilities
    ────────────────
    • Allocate nodes; place default start / goal markers.
    • Provide spatial queries: bounds check, neighbours, pixel→cell.
    • Map editing: toggle walls, move start/goal, randomise obstacles.
    • Dynamic obstacle spawning during agent movement.
    • Render all cells to a Pygame surface (separated from logic).
    """

    def __init__(self, rows: int, cols: int, rect: pygame.Rect) -> None:
        self.rows:      int         = rows
        self.cols:      int         = cols
        self.grid_rect: pygame.Rect = rect
        self.cell_size: int         = min(rect.width // cols, rect.height // rows)

        # 2-D node array
        self.nodes: List[List[Node]] = [
            [Node(r, c) for c in range(cols)]
            for r in range(rows)
        ]

        # Default positions: start = left-centre, goal = right-centre
        self.start: Node = self.nodes[rows // 2][cols // 4]
        self.start.is_start = True
        self.goal:  Node = self.nodes[rows // 2][(3 * cols) // 4]
        self.goal.is_goal = True

    # ── Spatial queries ───────────────────────────────────────────────────────

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_node(self, r: int, c: int) -> Node:
        return self.nodes[r][c]

    def get_neighbors(self, node: Node) -> List[Node]:
        """Return passable 4-connected neighbours of a node."""
        neighbors: List[Node] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = node.row + dr, node.col + dc
            if self.in_bounds(nr, nc):
                nb = self.nodes[nr][nc]
                if not nb.is_wall:
                    neighbors.append(nb)
        return neighbors

    def pixel_to_cell(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to (row, col); returns None if outside grid."""
        if not self.grid_rect.collidepoint(x, y):
            return None
        col = (x - self.grid_rect.left) // self.cell_size
        row = (y - self.grid_rect.top)  // self.cell_size
        return (row, col) if self.in_bounds(row, col) else None

    # ── State-clearing helpers ────────────────────────────────────────────────

    def clear_obstacles(self) -> None:
        """Remove all walls; preserve start and goal flags."""
        for row in self.nodes:
            for n in row:
                if not n.is_start and not n.is_goal:
                    n.is_wall = False

    def clear_search_state(self) -> None:
        """Reset search fields on every node (called on full agent reset)."""
        for row in self.nodes:
            for n in row:
                n.reset_search_state()

    def clear_paths_only(self) -> None:
        """Reset search state while keeping obstacle geometry intact."""
        for row in self.nodes:
            for n in row:
                n.reset_search_state()

    # ── Map-editing ───────────────────────────────────────────────────────────

    def randomize_obstacles(self, density: float) -> None:
        """Randomly place walls at approximately `density` fraction of cells."""
        self.clear_obstacles()
        for row in self.nodes:
            for n in row:
                if not n.is_start and not n.is_goal:
                    n.is_wall = random.random() < density

    def toggle_wall(self, r: int, c: int) -> None:
        """Flip wall state of a cell; no-op on start and goal."""
        n = self.get_node(r, c)
        if not n.is_start and not n.is_goal:
            n.is_wall = not n.is_wall

    def set_start(self, r: int, c: int) -> None:
        """Move the start marker to (r, c); rejected on walls and goal."""
        if not self.in_bounds(r, c):
            return
        n = self.get_node(r, c)
        if n.is_goal or n.is_wall:
            return
        self.start.is_start = False
        self.start = n
        n.is_start = True

    def set_goal(self, r: int, c: int) -> None:
        """Move the goal marker to (r, c); rejected on walls and start."""
        if not self.in_bounds(r, c):
            return
        n = self.get_node(r, c)
        if n.is_start or n.is_wall:
            return
        self.goal.is_goal = False
        self.goal = n
        n.is_goal = True

    # ── Dynamic obstacle spawning ─────────────────────────────────────────────

    def spawn_dynamic_obstacle(
        self,
        probability: float,
        forbidden: Set[Tuple[int, int]],
    ) -> Optional[Node]:
        """
        With `probability`, convert one random free cell into a wall.

        `forbidden` contains cells that must NOT become walls (the agent's
        current position, the remaining path, start, and goal).  This keeps
        the agent from being instantly trapped.

        Returns the newly-walled Node, or None if nothing was spawned.
        """
        if random.random() > probability:
            return None

        candidates = [
            n for row in self.nodes for n in row
            if not n.is_wall and not n.is_start and not n.is_goal
            and (n.row, n.col) not in forbidden
        ]
        if not candidates:
            return None

        chosen = random.choice(candidates)
        chosen.is_wall = True
        return chosen

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, pulse: float) -> None:
        """
        Render every cell to `surface`.

        `pulse` ∈ [0, 1] is driven by a sine wave so frontier / path
        cells appear to glow, giving clear visual feedback during search.
        """
        cs        = self.cell_size
        ox, oy    = self.grid_rect.left, self.grid_rect.top
        corner_r  = max(1, cs // 7)

        # Draw grid background (creates the thin-line grid effect)
        pygame.draw.rect(surface, COL_CELL_GRID_LN, self.grid_rect)

        for row in self.nodes:
            for n in row:
                lx = ox + n.col * cs + 1
                ty = oy + n.row * cs + 1
                cw = cs - 2
                cell_rect = pygame.Rect(lx, ty, cw, cw)

                # ── Colour priority (highest wins) ────────────────────────
                if   n.is_wall:        color = COL_CELL_WALL
                elif n.in_closed_set:  color = COL_CELL_VISITED   # blue
                elif n.in_open_set:    color = COL_CELL_FRONTIER  # yellow
                elif n.in_path:        color = COL_CELL_PATH      # green
                else:                  color = COL_CELL_EMPTY
                if n.is_start:         color = COL_CELL_START     # always visible
                if n.is_goal:          color = COL_CELL_GOAL      # always visible

                pygame.draw.rect(surface, color, cell_rect, border_radius=corner_r)

                # Animated glow overlay on frontier cells (yellow pulse)
                if n.in_open_set and not n.is_start and not n.is_goal and cs >= 8:
                    glow = pygame.Surface((cw, cw), pygame.SRCALPHA)
                    glow.fill((255, 215, 80, int(55 + 45 * pulse)))
                    surface.blit(glow, cell_rect.topleft)

                # Path shimmer (subtle green shimmer on solution path)
                if n.in_path and not n.is_start and not n.is_goal and cs >= 8:
                    shimmer = pygame.Surface((cw, cw), pygame.SRCALPHA)
                    shimmer.fill((80, 255, 160, int(25 + 20 * pulse)))
                    surface.blit(shimmer, cell_rect.topleft)

                # Wall micro-texture: a small centre dot
                if n.is_wall and cs >= 10:
                    pygame.draw.circle(
                        surface, (14, 14, 20),
                        cell_rect.center, max(1, cs // 8),
                    )

                # Start / goal: extra outline ring for clarity
                if n.is_start or n.is_goal:
                    ring_col = COL_CELL_START if n.is_start else COL_CELL_GOAL
                    pygame.draw.rect(
                        surface, ring_col, cell_rect,
                        max(1, cs // 10), border_radius=corner_r,
                    )


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── HEURISTIC FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def heuristic_manhattan(a: Node, b: Node) -> float:
    """Manhattan distance — admissible on 4-connected unit-cost grids."""
    return abs(a.row - b.row) + abs(a.col - b.col)


def heuristic_euclidean(a: Node, b: Node) -> float:
    """Euclidean straight-line distance."""
    return math.hypot(a.row - b.row, a.col - b.col)


# Lookup table avoids branching inside the hot search loop
HEURISTIC_MAP: Dict[HeuristicType, Callable[[Node, Node], float]] = {
    HeuristicType.MANHATTAN: heuristic_manhattan,
    HeuristicType.EUCLIDEAN: heuristic_euclidean,
}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── SEARCH ALGORITHMS  (base class + two concrete implementations)
# ═════════════════════════════════════════════════════════════════════════════

class SearchAlgorithm:
    """
    Abstract base for informed graph-search algorithms.

    The step-based interface (call `step()` once per frame) decouples
    the algorithm from the render loop, enabling smooth animation of
    node expansions without blocking the UI.

    Priority queue
    ──────────────
    Uses Python's `heapq` (binary min-heap).  A monotone tie-breaker
    integer avoids Node comparisons and ensures stable, FIFO ordering
    among equal-priority entries.

    Subclass contract
    ─────────────────
    Override `_evaluate(node)` to return the priority f(n):
      A*   → g(n) + h(n)
      GBFS → h(n)
    """

    def __init__(
        self,
        grid:      Grid,
        start:     Node,
        goal:      Node,
        heuristic: HeuristicType,
    ) -> None:
        self.grid         = grid
        self.start        = start
        self.goal         = goal
        self.heuristic_fn = HEURISTIC_MAP[heuristic]

        # Priority queue entries: (f_cost, tie_counter, Node)
        self.open_heap:   List[Tuple[float, int, Node]] = []
        self._tie_counter = 0

        self.status:         SearchStatus = SearchStatus.IDLE
        self.nodes_expanded: int          = 0
        self._t_start:       float        = 0.0
        self._t_end:         float        = 0.0

        self._initialise()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _push(self, node: Node, priority: float) -> None:
        """Push a node onto the min-heap with a stable tie-breaker."""
        self._tie_counter += 1
        heapq.heappush(self.open_heap, (priority, self._tie_counter, node))
        node.in_open_set = True

    def _pop(self) -> Node:
        """Pop and return the lowest-priority node."""
        _, _, node = heapq.heappop(self.open_heap)
        return node

    def _initialise(self) -> None:
        """Seed the open list with the start node."""
        self.grid.clear_paths_only()
        self.open_heap.clear()
        self._tie_counter    = 0
        self.nodes_expanded  = 0
        self.status          = SearchStatus.RUNNING
        self._t_start        = time.perf_counter()
        self._t_end          = 0.0

        # Start node: g=0, h=h(start,goal), f=_evaluate(start)
        self.start.g_cost = 0.0
        self.start.h_cost = self.heuristic_fn(self.start, self.goal)
        self.start.f_cost = self._evaluate(self.start)
        self._push(self.start, self.start.f_cost)

    def _evaluate(self, node: Node) -> float:
        """
        f(n) — priority function.  Must be overridden by subclasses.
          A*   → return node.g_cost + node.h_cost
          GBFS → return node.h_cost
        """
        raise NotImplementedError

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self) -> SearchStatus:
        """
        Perform one node expansion.

        Called STEPS_PER_FRAME times per render frame so the GUI can
        animate the search without any blocking sleep.

        Returns the current SearchStatus; callers should stop calling
        once SUCCESS or FAILURE is returned.
        """
        if self.status is not SearchStatus.RUNNING:
            return self.status

        # Open list exhausted → no solution
        if not self.open_heap:
            self.status = SearchStatus.FAILURE
            self._t_end = time.perf_counter()
            return self.status

        # Pop lowest-cost node
        current = self._pop()
        current.in_open_set   = False
        current.in_closed_set = True
        self.nodes_expanded  += 1

        # Goal test
        if current is self.goal:
            self.status = SearchStatus.SUCCESS
            self._t_end = time.perf_counter()
            self._reconstruct_path()
            return self.status

        # Expand: evaluate each passable neighbour
        for nb in self.grid.get_neighbors(current):
            tentative_g = current.g_cost + 1.0   # uniform edge cost

            if tentative_g < nb.g_cost:
                nb.parent = current
                nb.g_cost = tentative_g
                nb.h_cost = self.heuristic_fn(nb, self.goal)
                nb.f_cost = self._evaluate(nb)

                # Lazy re-insertion: push updated entry; stale entries
                # are silently skipped when popped (in_closed_set check)
                if not nb.in_closed_set:
                    self._push(nb, nb.f_cost)

        return self.status

    def _reconstruct_path(self) -> None:
        """Walk parent pointers from goal → start and set in_path = True."""
        node: Optional[Node] = self.goal
        while node is not None:
            node.in_path = True
            node = node.parent

    @property
    def path(self) -> List[Node]:
        """Ordered path list [start, …, goal]; empty if search not succeeded."""
        if self.status is not SearchStatus.SUCCESS:
            return []
        result: List[Node] = []
        node: Optional[Node] = self.goal
        while node is not None:
            result.append(node)
            node = node.parent
        result.reverse()
        return result

    @property
    def execution_time_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds (live while still running)."""
        end = self._t_end if self._t_end > 0.0 else time.perf_counter()
        return (end - self._t_start) * 1000.0


class AStarSearch(SearchAlgorithm):
    """
    A* Search — f(n) = g(n) + h(n).

    Optimal when h is admissible (never over-estimates).
    Manhattan distance is admissible on 4-connected unit-cost grids.
    """
    def _evaluate(self, node: Node) -> float:
        return node.g_cost + node.h_cost   # f = g + h


class GreedyBestFirstSearch(SearchAlgorithm):
    """
    Greedy Best-First Search — f(n) = h(n).

    Expands only toward the goal; fast but not guaranteed optimal.
    """
    def _evaluate(self, node: Node) -> float:
        return node.h_cost   # f = h only


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── AGENT  (search orchestrator + path follower)
# ═════════════════════════════════════════════════════════════════════════════

class Agent:
    """
    The pathfinding agent.

    Drives the search algorithm frame-by-frame (SEARCHING state), then
    animates movement along the discovered path (MOVING state).

    Dynamic re-planning  (§4)
    ─────────────────────────
    When dynamic mode is ON, the Grid may spawn new walls each frame.
    If a new wall falls on the agent's remaining path, `handle_dynamic_obstacle()`
    is called, which triggers `_replan_from_current()` — re-planning starts
    from the agent's *current position*, NOT the original start, so only the
    affected portion is re-searched (optimised, as required by §4).

    Rendering
    ─────────
    `draw()` accepts a Pygame surface so rendering stays decoupled from logic.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid              = grid
        self.state             = AgentState.IDLE
        self.algorithm_type    = AlgorithmType.ASTAR
        self.heuristic_type    = HeuristicType.MANHATTAN
        self.dynamic_mode      = False

        self.current_node:  Node          = grid.start
        self.current_path:  List[Node]    = []
        self.path_index:    int           = 0
        self.search: Optional[SearchAlgorithm] = None
        self._last_move_time: float       = 0.0

        # Live metrics (displayed in the GUI dashboard)
        self.metric_nodes_expanded: int   = 0
        self.metric_path_cost:      float = 0.0
        self.metric_exec_ms:        float = 0.0

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_algorithm(self, algo: AlgorithmType) -> None:
        self.algorithm_type = algo

    def set_heuristic(self, heuristic: HeuristicType) -> None:
        self.heuristic_type = heuristic

    def toggle_dynamic_mode(self) -> None:
        self.dynamic_mode = not self.dynamic_mode

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Return to IDLE; clear visualisation but keep obstacle layout."""
        self.state             = AgentState.IDLE
        self.grid.clear_search_state()
        self.current_node      = self.grid.start
        self.current_path      = []
        self.path_index        = 0
        self.search            = None
        self.metric_nodes_expanded = 0
        self.metric_path_cost      = 0.0
        self.metric_exec_ms        = 0.0

    def start_search(self) -> None:
        """Begin a fresh search from the grid's current start node."""
        self.grid.clear_paths_only()
        self.current_node  = self.grid.start
        self.current_path  = []
        self.path_index    = 0
        self.search        = self._build_search(self.grid.start)
        self.state         = AgentState.SEARCHING

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_search(self, from_node: Node) -> SearchAlgorithm:
        """Instantiate the selected algorithm class from `from_node`."""
        if self.algorithm_type == AlgorithmType.ASTAR:
            return AStarSearch(self.grid, from_node, self.grid.goal, self.heuristic_type)
        return GreedyBestFirstSearch(self.grid, from_node, self.grid.goal, self.heuristic_type)

    def _replan_from_current(self) -> None:
        """
        Re-plan path from the agent's current position after a dynamic
        obstacle blocks the route.  Does NOT restart from the original start
        node — only the remaining portion of the journey is re-searched.
        This is the optimised re-planning required by §4.
        """
        self.grid.clear_paths_only()
        self.search = self._build_search(self.current_node)
        self.state  = AgentState.SEARCHING

    def _capture_metrics(self) -> None:
        """Snapshot search statistics when a path is found."""
        if self.search is None:
            return
        self.metric_nodes_expanded = self.search.nodes_expanded
        self.metric_path_cost      = max(0.0, len(self.current_path) - 1)
        self.metric_exec_ms        = self.search.execution_time_ms

    # ── Frame update ──────────────────────────────────────────────────────────

    def update(self) -> None:
        """
        Advance the agent's state machine by one frame.

        SEARCHING → run up to STEPS_PER_FRAME algorithm steps.
        MOVING    → advance one cell along the path every AGENT_MOVE_DELAY s.
        """
        if self.state == AgentState.SEARCHING and self.search is not None:
            for _ in range(STEPS_PER_FRAME):
                status = self.search.step()

                if status is SearchStatus.SUCCESS:
                    self.current_path = self.search.path
                    self.path_index   = 0
                    self.state        = AgentState.MOVING
                    if self.current_path:
                        self.current_node = self.current_path[0]
                    self._capture_metrics()
                    self._last_move_time = time.perf_counter()
                    break

                if status is SearchStatus.FAILURE:
                    self.state                 = AgentState.NO_PATH
                    self.metric_nodes_expanded = self.search.nodes_expanded
                    self.metric_path_cost      = float("inf")
                    self.metric_exec_ms        = self.search.execution_time_ms
                    break

        elif self.state == AgentState.MOVING and self.current_path:
            now = time.perf_counter()
            if now - self._last_move_time >= AGENT_MOVE_DELAY:
                self._last_move_time = now
                if self.path_index < len(self.current_path) - 1:
                    self.path_index  += 1
                    self.current_node = self.current_path[self.path_index]
                    # Defensive: re-plan if agent steps onto a wall
                    if self.current_node.is_wall:
                        self._replan_from_current()
                else:
                    # Arrived at goal
                    self.state        = AgentState.IDLE
                    self.current_node = self.grid.goal

    # ── Dynamic obstacle response ─────────────────────────────────────────────

    def handle_dynamic_obstacle(self, spawned: Optional[Node]) -> None:
        """
        React to a newly spawned obstacle.

        Re-plans only if the obstacle is on the remaining path segment
        (§4: "Do NOT restart entire search if obstacle is not on current path").
        """
        if (
            self.state is not AgentState.MOVING
            or spawned is None
            or not self.current_path
        ):
            return

        remaining_positions = {
            (n.row, n.col) for n in self.current_path[self.path_index:]
        }
        if (spawned.row, spawned.col) in remaining_positions:
            self._replan_from_current()

    def get_forbidden_cells(self) -> Set[Tuple[int, int]]:
        """
        Cells that dynamic obstacles must not overwrite.

        Includes start, goal, agent's current position, and remaining
        path cells to avoid constant trivial re-plans.
        """
        forbidden: Set[Tuple[int, int]] = {
            (self.grid.start.row,       self.grid.start.col),
            (self.grid.goal.row,        self.grid.goal.col),
            (self.current_node.row,     self.current_node.col),
        }
        for n in self.current_path[self.path_index:]:
            forbidden.add((n.row, n.col))
        return forbidden

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, grid: Grid, pulse: float) -> None:
        """Render the agent as an animated cyan rectangle over its current cell."""
        n  = self.current_node
        cs = grid.cell_size
        lx = grid.grid_rect.left + n.col * cs + 1
        ty = grid.grid_rect.top  + n.row * cs + 1

        pad  = cs * 0.14
        rect = pygame.Rect(lx + pad, ty + pad, cs - pad * 2 - 2, cs - pad * 2 - 2)
        rr   = max(2, cs // 5)

        # Outer pulsing halo (alpha surface)
        halo_expand = int(cs * 0.20 * (1.0 + 0.40 * pulse))
        halo_surf   = pygame.Surface(
            (rect.w + halo_expand * 2, rect.h + halo_expand * 2),
            pygame.SRCALPHA,
        )
        halo_surf.fill((*COL_CELL_AGENT, int(35 + 28 * pulse)))
        surface.blit(halo_surf, (rect.left - halo_expand, rect.top - halo_expand))

        # Solid agent body
        pygame.draw.rect(surface, COL_CELL_AGENT, rect, border_radius=rr)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── UI COMPONENTS  (Button + drawing helpers)
# ═════════════════════════════════════════════════════════════════════════════

def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t:  float,
) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colours."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def draw_rounded_rect(
    surface:      pygame.Surface,
    color:        Tuple[int, int, int],
    rect:         pygame.Rect,
    radius:       int = 6,
    border_width: int = 0,
    border_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Fill a rounded rectangle, then optionally draw a rounded border."""
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_width and border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)


def draw_hline(
    surface: pygame.Surface,
    x1: int, y: int, x2: int,
    color: Tuple[int, int, int] = COL_BORDER,
) -> None:
    pygame.draw.line(surface, color, (x1, y), (x2, y))


class Button:
    """
    Flat, animated panel button.

    Supports
    ────────
    • Smooth hover colour lerp (driven by delta-time, 16 Hz blend).
    • Toggle mode: stays visually "active" after click until clicked again.
    • Accent colour tints the border and label text when active.
    • Small dot indicator on the right edge when active.
    """

    def __init__(
        self,
        rect:      pygame.Rect,
        label:     str,
        font:      pygame.font.Font,
        is_toggle: bool = False,
        accent:    Tuple[int, int, int] = COL_BLUE,
        icon:      str = "",
    ) -> None:
        self.rect      = rect
        self.label     = label
        self.font      = font
        self.is_toggle = is_toggle
        self.accent    = accent
        self.icon      = icon
        self.active    = False
        self._hover    = False
        self._anim_t   = 0.0   # hover progress ∈ [0, 1]

    def update(self, dt: float) -> None:
        """Advance hover animation — must be called every frame."""
        target       = 1.0 if self._hover else 0.0
        self._anim_t += (target - self._anim_t) * min(1.0, dt * 16)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Update internal state from a Pygame event.
        Returns True when the button is clicked (triggers the action).
        """
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.active = not self.active
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        t = self._anim_t

        # Background: lerp toward hover tint; darken with accent when active
        bg = lerp_color(COL_BTN_BG, COL_BTN_HOVER, t)
        if self.active:
            tint = tuple(max(0, v // 7) for v in self.accent)   # type: ignore[arg-type]
            bg   = lerp_color(COL_BTN_BG, tint, 0.9)            # type: ignore[arg-type]

        # Border: lerp toward accent
        bord_col = lerp_color(
            COL_BORDER, self.accent,
            0.55 if self.active else t * 0.4,
        )
        bord_w = 2 if self.active else 1

        draw_rounded_rect(surface, bg, self.rect, radius=7)
        draw_rounded_rect(
            surface, COL_CARD_BG, self.rect,
            radius=7, border_width=bord_w, border_color=bord_col,
        )

        # Label colour
        if self.active:
            txt_col = lerp_color(COL_TEXT1, self.accent, 0.85)
        else:
            txt_col = lerp_color(COL_TEXT2, COL_TEXT1, t)

        full_label = (self.icon + "  " if self.icon else "") + self.label
        ts         = self.font.render(full_label, True, txt_col)
        surface.blit(ts, ts.get_rect(center=self.rect.center))

        # Active-state dot indicator (right edge)
        if self.active:
            pygame.draw.circle(
                surface, self.accent,
                (self.rect.right - 11, self.rect.centery), 4,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── APPLICATION  (main window, event loop, rendering)
# ═════════════════════════════════════════════════════════════════════════════

class App:
    """
    Top-level application controller.

    Owns the Pygame window, the Grid, and the Agent.  The main loop
    is divided into three clean phases — matching the standard game-loop
    pattern and satisfying the "separate logic from rendering" requirement:

        handle_events()  →  pure input handling, no state side-effects
        update()         →  simulation / animation tick
        draw()           →  pure rendering, reads state but never writes it
    """

    def __init__(self, rows: int = 28, cols: int = 42) -> None:
        pygame.init()
        pygame.display.set_caption("PathViz  ·  Dynamic Pathfinding Agent")
        self.screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock   = pygame.time.Clock()

        self._load_fonts()

        self.grid    = Grid(rows, cols, GRID_RECT)
        self.agent   = Agent(self.grid)
        self.density = DEFAULT_DENSITY
        self.running = True

        # Time / animation bookkeeping
        self._dt:      float = 0.016
        self._elapsed: float = 0.0
        self._pulse:   float = 0.0   # sine-wave ∈ [0, 1] for cell glow effects

        # One-line status toast at the bottom of the panel
        self._notif_text:  str   = ""
        self._notif_color: Tuple[int,int,int] = COL_TEXT2
        self._notif_timer: float = 0.0

        self._build_ui()

    # ── Font loading ──────────────────────────────────────────────────────────

    def _load_fonts(self) -> None:
        """Try a cascade of monospace fonts; fall back to Pygame's default."""
        preferred = (
            "Cascadia Code", "JetBrains Mono",
            "Consolas", "Lucida Console", "Courier New",
        )
        for name in preferred:
            try:
                pygame.font.SysFont(name, 10)   # existence probe
                self.f_xs = pygame.font.SysFont(name, 13)
                self.f_sm = pygame.font.SysFont(name, 15)
                self.f_md = pygame.font.SysFont(name, 17)
                self.f_lg = pygame.font.SysFont(name, 19, bold=True)
                self.f_xl = pygame.font.SysFont(name, 24, bold=True)
                return
            except Exception:
                pass
        # Absolute fallback — always succeeds
        for attr, sz, bold in [
            ("f_xs", 13, False), ("f_sm", 15, False), ("f_md", 17, False),
            ("f_lg", 19, True),  ("f_xl", 24, True),
        ]:
            setattr(self, attr, pygame.font.SysFont(None, sz))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Instantiate all panel Button objects."""
        px = WINDOW_W - PANEL_W + 10
        bw = PANEL_W - 20

        def make_btn(y, label, icon="", toggle=False, accent=COL_BLUE) -> Button:
            return Button(
                pygame.Rect(px, y, bw, 34), label, self.f_sm,
                is_toggle=toggle, accent=accent, icon=icon,
            )

        # Buttons start at y=72 (below header block), spaced 42px apart
        self.btn_start = make_btn(72,  "Start Search",       icon="▶", accent=COL_GREEN)
        self.btn_algo  = make_btn(114, "Algorithm:  A*",     icon="⚙")
        self.btn_heur  = make_btn(156, "Heuristic:  Manhattan")
        self.btn_dyn   = make_btn(198, "Dynamic Mode:  OFF", icon="⚡",
                                  toggle=True, accent=COL_AMBER)
        self.btn_maze  = make_btn(252, "Randomize Maze",     icon="⟳")
        self.btn_clear = make_btn(294, "Clear Obstacles",    icon="✕", accent=COL_RED)

        self._all_buttons: List[Button] = [
            self.btn_start, self.btn_algo, self.btn_heur,
            self.btn_dyn,   self.btn_maze, self.btn_clear,
        ]
        self._button_keys: List[str] = [
            "start", "algo", "heur", "dyn", "maze", "clear",
        ]

    # ── Notification helper ───────────────────────────────────────────────────

    def _notify(self, text: str, color: Tuple[int, int, int]) -> None:
        """Display a short toast message at the bottom of the panel for ~2.5 s."""
        self._notif_text  = text
        self._notif_color = color
        self._notif_timer = 2.5

    # ── Button action dispatcher ──────────────────────────────────────────────

    def _on_button(self, key: str) -> None:
        """Handle every button action in one central method."""
        if key == "start":
            self.agent.start_search()
            self._notify("Search started", COL_GREEN)

        elif key == "algo":
            if self.agent.algorithm_type == AlgorithmType.ASTAR:
                self.agent.set_algorithm(AlgorithmType.GBFS)
                self.btn_algo.label = "Algorithm:  Greedy BFS"
            else:
                self.agent.set_algorithm(AlgorithmType.ASTAR)
                self.btn_algo.label = "Algorithm:  A*"
            self.agent.reset()

        elif key == "heur":
            if self.agent.heuristic_type == HeuristicType.MANHATTAN:
                self.agent.set_heuristic(HeuristicType.EUCLIDEAN)
                self.btn_heur.label = "Heuristic:  Euclidean"
            else:
                self.agent.set_heuristic(HeuristicType.MANHATTAN)
                self.btn_heur.label = "Heuristic:  Manhattan"
            self.agent.reset()

        elif key == "dyn":
            self.agent.toggle_dynamic_mode()
            self.btn_dyn.label = (
                "Dynamic Mode:  ON "
                if self.agent.dynamic_mode else
                "Dynamic Mode:  OFF"
            )
            self._notify(
                "Dynamic obstacles ON" if self.agent.dynamic_mode
                else "Dynamic obstacles OFF",
                COL_AMBER,
            )

        elif key == "maze":
            self.agent.reset()
            self.grid.randomize_obstacles(self.density)
            self._notify(f"Maze generated  ({int(self.density * 100)}% density)", COL_AMBER)

        elif key == "clear":
            self.agent.reset()
            self.grid.clear_obstacles()
            self._notify("Obstacles cleared", COL_TEXT2)

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            # Grid and panel both receive every event
            self._handle_grid_mouse(event)
            for btn, key in zip(self._all_buttons, self._button_keys):
                if btn.handle_event(event):
                    self._on_button(key)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        k = event.key
        if   k == pygame.K_ESCAPE:       self.running = False
        elif k == pygame.K_SPACE:        self._on_button("start")
        elif k == pygame.K_r:            self.agent.reset()
        elif k == pygame.K_c:
            self.agent.reset()
            self.grid.clear_obstacles()
        elif k == pygame.K_LEFTBRACKET:
            self.density = max(MIN_DENSITY, round(self.density - 0.05, 2))
        elif k == pygame.K_RIGHTBRACKET:
            self.density = min(MAX_DENSITY, round(self.density + 0.05, 2))

    def _handle_grid_mouse(self, event: pygame.event.Event) -> None:
        """Translate a grid left-click into wall / start / goal edits."""
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        cell = self.grid.pixel_to_cell(*event.pos)
        if cell is None:
            return
        r, c = cell
        mods = pygame.key.get_mods()

        if mods & pygame.KMOD_SHIFT:
            self.agent.reset()
            self.grid.set_start(r, c)
            self.agent.current_node = self.grid.start
        elif mods & pygame.KMOD_CTRL:
            self.agent.reset()
            self.grid.set_goal(r, c)
        else:
            self.agent.reset()
            self.grid.toggle_wall(r, c)

    # ── Simulation update ─────────────────────────────────────────────────────

    def update(self) -> None:
        """Advance all simulation and UI animation by one frame."""
        dt             = self._dt
        self._elapsed += dt
        self._pulse    = (math.sin(self._elapsed * 3.6) + 1.0) / 2.0  # ∈ [0,1]

        for btn in self._all_buttons:
            btn.update(dt)

        # Attempt to spawn a dynamic obstacle (no-op if mode is OFF)
        spawned: Optional[Node] = None
        if self.agent.dynamic_mode and self.agent.state == AgentState.MOVING:
            spawned = self.grid.spawn_dynamic_obstacle(
                DYN_OBSTACLE_PROB, self.agent.get_forbidden_cells()
            )

        # Let the agent react to the new obstacle (may trigger re-plan)
        self.agent.handle_dynamic_obstacle(spawned)
        self.agent.update()

        if self._notif_timer > 0:
            self._notif_timer -= dt

    # ── Rendering pipeline ────────────────────────────────────────────────────

    def draw(self) -> None:
        """Full-frame render: canvas → grid glow → grid → agent → panel."""
        self.screen.fill(COL_BG)

        # Subtle glow border around the grid area
        draw_rounded_rect(self.screen, COL_BORDER_HI, GRID_RECT.inflate(3, 3), radius=4)
        draw_rounded_rect(self.screen, COL_BG, GRID_RECT, radius=2)

        self.grid.draw(self.screen, self._pulse)

        if self.agent.state in (AgentState.MOVING, AgentState.IDLE, AgentState.NO_PATH):
            self.agent.draw(self.screen, self.grid, self._pulse)

        self._draw_panel()
        pygame.display.flip()

    # ── Panel rendering ───────────────────────────────────────────────────────

    def _draw_panel(self) -> None:
        """Render the entire right-side control, metrics, legend, and keybind panel."""
        px  = WINDOW_W - PANEL_W
        lx  = px + 12                 # left content x (generous padding)
        rx  = WINDOW_W - 12           # right content x
        pw  = rx - lx                 # usable interior width

        # ── Panel background ──────────────────────────────────────────────────
        pygame.draw.rect(self.screen, COL_PANEL_BG,
                         pygame.Rect(px, 0, PANEL_W, WINDOW_H))

        # Top accent bar (full-width coloured strip at top)
        pygame.draw.rect(self.screen, COL_BLUE,
                         pygame.Rect(px, 0, PANEL_W, 3))

        # Left border line
        pygame.draw.rect(self.screen, COL_BORDER_HI,
                         pygame.Rect(px, 0, 1, WINDOW_H))

        # ── 1. HEADER ─────────────────────────────────────────────────────────
        # Title
        self._put_text("PathViz", lx, 12, self.f_xl, COL_BLUE)
        # Subtitle on its own clearly separated line
        self._put_text("Dynamic Pathfinding Agent", lx, 42, self.f_xs, COL_TEXT2)
        draw_hline(self.screen, lx, 64, rx, COL_BORDER_HI)

        # ── 2. CONTROLS ───────────────────────────────────────────────────────
        self._section_label("CONTROLS", lx, 67)
        for btn in self._all_buttons:
            btn.draw(self.screen)

        # Density row (sits below last button — btn_clear ends at y=294+34=328)
        bar_y = 340
        self._put_text("Obstacle Density", lx, bar_y, self.f_xs, COL_TEXT2)
        pct_s = self.f_sm.render(f"{int(self.density * 100)}%", True, COL_AMBER)
        self.screen.blit(pct_s, (rx - pct_s.get_width(), bar_y - 1))
        bar = pygame.Rect(lx, bar_y + 20, pw, 7)
        draw_rounded_rect(self.screen, COL_CARD_BG2, bar, radius=4)
        fw = int(bar.w * (self.density - MIN_DENSITY) / (MAX_DENSITY - MIN_DENSITY))
        if fw > 0:
            draw_rounded_rect(self.screen, COL_AMBER,
                              pygame.Rect(bar.x, bar.y, max(fw, bar.h), bar.h),
                              radius=4)
        draw_rounded_rect(self.screen, COL_CARD_BG2, bar, radius=4,
                          border_width=1, border_color=COL_BORDER_HI)

        sep1 = bar.bottom + 12
        draw_hline(self.screen, lx, sep1, rx, COL_BORDER_HI)

        # ── 3. METRICS DASHBOARD ──────────────────────────────────────────────
        met_top = sep1 + 4
        self._section_label("METRICS", lx, met_top)

        state_color = {
            AgentState.IDLE:      COL_TEXT2,
            AgentState.SEARCHING: COL_AMBER,
            AgentState.MOVING:    COL_GREEN,
            AgentState.NO_PATH:   COL_RED,
        }.get(self.agent.state, COL_TEXT2)

        state_label = {
            AgentState.IDLE:      "Idle",
            AgentState.SEARCHING: "Searching…",
            AgentState.MOVING:    "Traversing",
            AgentState.NO_PATH:   "No Path",
        }.get(self.agent.state, "")

        # Metrics card
        card_top = met_top + 18
        card     = pygame.Rect(lx, card_top, pw, 114)
        draw_rounded_rect(self.screen, COL_CARD_BG, card, radius=10,
                          border_width=1, border_color=COL_BORDER_HI)

        # Status pill badge inside the card (top-right)
        badge_w  = max(88, self.f_xs.size(state_label)[0] + 20)
        badge    = pygame.Rect(card.right - badge_w - 8, card.top + 8,
                               badge_w, 20)
        draw_rounded_rect(self.screen, COL_CARD_BG2, badge,
                          radius=10, border_width=1, border_color=state_color)
        bs = self.f_xs.render(state_label, True, state_color)
        self.screen.blit(bs, bs.get_rect(center=badge.center))

        # Metric rows — label left, value right, good vertical spacing
        inf_cost = self.agent.metric_path_cost == float("inf")
        cost_str = "∞" if inf_cost else f"{self.agent.metric_path_cost:.0f}"
        cost_col = COL_RED if inf_cost else COL_GREEN

        metrics_rows = [
            ("Nodes Expanded", str(self.agent.metric_nodes_expanded), COL_TEXT1),
            ("Path Cost",      cost_str,                               cost_col),
            ("Exec Time",      f"{self.agent.metric_exec_ms:.2f} ms",  COL_BLUE),
        ]
        my = card.top + 36
        for lbl, val, vc in metrics_rows:
            ls = self.f_xs.render(lbl, True, COL_TEXT2)
            vs = self.f_lg.render(val, True, vc)          # larger value text
            self.screen.blit(ls, (card.left + 12, my))
            self.screen.blit(vs, (card.right - vs.get_width() - 12, my - 2))
            # thin divider between rows (except after last)
            if my < card.top + 36 + 2 * 26:
                pygame.draw.line(self.screen, COL_BORDER,
                                 (card.left + 10, my + 20),
                                 (card.right - 10, my + 20))
            my += 26

        sep2 = card.bottom + 12
        draw_hline(self.screen, lx, sep2, rx, COL_BORDER_HI)

        # ── 4. LEGEND ─────────────────────────────────────────────────────────
        leg_top = sep2 + 4
        self._section_label("LEGEND", lx, leg_top)

        # Each entry: (fill_color, border_color, label, description_suffix)
        # Wall uses a lighter display color so the swatch is visible on dark bg
        legend = [
            (COL_CELL_START,    (100, 180, 255), "Start",    "Blue  — origin node"),
            (COL_CELL_GOAL,     (180, 120, 255), "Goal",     "Violet — target node"),
            (COL_CELL_FRONTIER, (255, 200,  60), "Frontier", "Yellow — open set"),
            (COL_CELL_VISITED,  ( 60, 120, 210), "Visited",  "Blue  — closed set"),
            (COL_CELL_PATH,     ( 60, 210, 110), "Path",     "Green — solution"),
            (COL_CELL_AGENT,    ( 60, 230, 210), "Agent",    "Cyan  — current pos"),
            (( 55,  58,  75),   ( 90,  95, 120), "Wall",     "Dark  — obstacle"),
        ]

        SW   = 18    # swatch size in pixels
        RH   = 26    # row height
        gy   = leg_top + 20

        for fill, border, name, desc in legend:
            # ── Swatch (solid fill + bright border so it pops) ───────────────
            sw_rect = pygame.Rect(lx, gy, SW, SW)

            # Dark background behind swatch so even dark fills are visible
            bg_rect = pygame.Rect(lx - 2, gy - 2, SW + 4, SW + 4)
            pygame.draw.rect(self.screen, (38, 42, 58), bg_rect, border_radius=5)

            # Fill
            pygame.draw.rect(self.screen, fill, sw_rect, border_radius=4)
            # Bright border — always visible regardless of fill darkness
            pygame.draw.rect(self.screen, border, sw_rect, 2, border_radius=4)

            # ── Name (bright, bold-style) ────────────────────────────────────
            self._put_text(name, lx + SW + 8, gy + 2, self.f_sm, COL_TEXT1)

            # ── Description suffix (dim, right-aligned) ──────────────────────
            ds = self.f_xs.render(desc, True, COL_TEXT2)
            self.screen.blit(ds, (rx - ds.get_width(), gy + 4))

            # Thin row separator
            pygame.draw.line(self.screen, COL_BORDER,
                             (lx, gy + RH - 2), (rx, gy + RH - 2))
            gy += RH

        sep3 = gy + 4
        draw_hline(self.screen, lx, sep3, rx, COL_BORDER_HI)

        # ── 5. KEYBINDS ───────────────────────────────────────────────────────
        kb_top = sep3 + 4
        self._section_label("KEYBINDS", lx, kb_top)

        keybinds = [
            ("Space",      "start search"),
            ("R  /  C",    "reset  /  clear"),
            ("[ / ]",      "density ±5%"),
            ("LMB",        "toggle wall"),
            ("Shift+LMB",  "set start node"),
            ("Ctrl+LMB",   "set goal node"),
        ]
        kx_key  = lx
        kx_desc = lx + 100          # description column aligned
        ky      = kb_top + 18
        row_gap = 19
        for kb, desc in keybinds:
            self._put_text(kb,   kx_key,  ky, self.f_xs, COL_BLUE)
            self._put_text(desc, kx_desc, ky, self.f_xs, COL_TEXT1)
            ky += row_gap

        # ── Notification toast (bottom centre of panel) ───────────────────────
        if self._notif_timer > 0 and self._notif_text:
            alpha = min(1.0, self._notif_timer)
            ns  = self.f_xs.render(self._notif_text, True, self._notif_color)
            # pill background for the toast
            toast_rect = pygame.Rect(
                px + (PANEL_W - ns.get_width() - 20) // 2,
                WINDOW_H - 30,
                ns.get_width() + 20, 20,
            )
            draw_rounded_rect(self.screen, COL_CARD_BG, toast_rect, radius=10,
                              border_width=1, border_color=self._notif_color)
            self.screen.blit(ns, (toast_rect.x + 10, toast_rect.y + 3))

        # ── Live FPS (bottom-right corner) ────────────────────────────────────
        fps_s = self.f_xs.render(f"{self.clock.get_fps():.0f} fps", True, COL_TEXT3)
        self.screen.blit(fps_s, (rx - fps_s.get_width(), WINDOW_H - 16))

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def _put_text(
        self,
        text:  str,
        x:     int,
        y:     int,
        font:  pygame.font.Font,
        color: Tuple[int, int, int],
    ) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _section_label(self, text: str, x: int, y: int) -> None:
        """Render an uppercase section heading — clearly readable."""
        self.screen.blit(self.f_xs.render(text, True, (110, 120, 155)), (x, y))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Enter the 60-fps event / update / draw loop."""
        while self.running:
            self._dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update()
            self.draw()
        pygame.quit()


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ── ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def prompt_grid_size() -> Tuple[int, int]:
    """
    Ask for grid dimensions at startup.
    Press Enter to accept defaults (28 rows × 42 cols).
    Values are clamped to [5, 80] for both axes.
    """
    print()
    print("  ╔════════════════════════════════════════════╗")
    print("  ║   PathViz — Dynamic Pathfinding Agent     ║")
    print("  ║   AI Course  ·  Informed Search Project   ║")
    print("  ╚════════════════════════════════════════════╝")
    print()
    print("  Window: 1280 × 680   |   Suggested: 28 rows × 42 cols")
    print("  Press Enter to accept the default value shown in [ ]")
    print()

    def read_int(prompt: str, default: int) -> int:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            return max(5, min(80, int(raw))) if raw else default
        except (ValueError, EOFError):
            return default

    rows = read_int("Grid rows", 28)
    cols = read_int("Grid cols", 42)
    print()
    return rows, cols


def main() -> None:
    rows, cols = prompt_grid_size()
    App(rows, cols).run()


if __name__ == "__main__":
    main()
