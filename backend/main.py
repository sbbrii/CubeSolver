from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Dict, Any
import time
from fastapi.staticfiles import StaticFiles
from pathlib import Path


class SolveRequest(BaseModel):
    # Flat 54-length list of facelets in URFDLB order faces, each with color char
    state: List[str]
    trace: bool = False


class SolveResponse(BaseModel):
    moves: List[str]
    explored_states: int
    time_ms: int
    depth: int
    meta: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] | None = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parents[1] / "frontend" / "static"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------- Cube model (simplified 3x3 representation) ----------
# We use a facelet model: 54 characters (6 faces x 9 facelets)
# Faces in order: U, R, F, D, L, B. Colors are arbitrary single letters.


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def is_goal(state: List[str]) -> bool:
    if len(state) != 54:
        return False
    for i in range(0, 54, 9):
        face = state[i : i + 9]
        if len(set(face)) != 1:
            return False
    return True


# Moves: For demo, we define a minimal subset and permutations on indices.
# In a full solver you would define all 18 quarter/half turns.


Move = Literal[
    "U", "U'", "U2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "B", "B'", "B2",
]


def apply_move(state: List[str], move: Move) -> List[str]:
    # Facelet indices by face: U(0-8), R(9-17), F(18-26), D(27-35), L(36-44), B(45-53)
    s = state.copy()

    def rot_face(face_start: int, prime: bool = False, half: bool = False):
        # rotate 3x3 face indices 0..8 around center (clockwise)
        idx = [face_start + i for i in range(9)]
        a = s.copy()
        if half:
            mapping = {0:8, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, 8:0}
        elif not prime:
            mapping = {0:6, 1:3, 2:0, 3:7, 4:4, 5:1, 6:8, 7:5, 8:2}
        else:
            mapping = {0:2, 1:5, 2:8, 3:1, 4:4, 5:7, 6:0, 7:3, 8:6}
        for i in range(9):
            s[idx[i]] = a[idx[mapping[i]]]

    def cycle(strips: list[list[int]]):
        # cycle strips forward: 0->1->2->3->0
        a = s.copy()
        for i in range(4):
            src = strips[(i + 3) % 4]
            dst = strips[i]
            for j in range(3):
                s[dst[j]] = a[src[j]]

    def move_U(prime=False, half=False):
        rot_face(0, prime, half)
        topF = [18 + i for i in [0,1,2]]
        topR = [9 + i for i in [0,1,2]]
        topB = [45 + i for i in [0,1,2]]
        topL = [36 + i for i in [0,1,2]]
        if half:
            cycle([topF, topB, topR, topL])
            cycle([topF, topB, topR, topL])
        elif not prime:
            cycle([topF, topR, topB, topL])
        else:
            cycle([topF, topL, topB, topR])

    def move_R(prime=False, half=False):
        rot_face(9, prime, half)
        u = [0 + i for i in [2,5,8]]
        f = [18 + i for i in [2,5,8]]
        d = [27 + i for i in [2,5,8]]
        b = [45 + i for i in [6,3,0]]  # left column of B (note orientation)
        if half:
            cycle([u, d, f, b]); cycle([u, d, f, b])
        elif not prime:
            cycle([u, f, d, b])
        else:
            cycle([u, b, d, f])

    def move_F(prime=False, half=False):
        rot_face(18, prime, half)
        u = [0 + i for i in [6,7,8]]
        r = [9 + i for i in [0,3,6]]
        d = [27 + i for i in [2,1,0]]
        l = [36 + i for i in [8,5,2]]
        if half:
            cycle([u, d, r, l]); cycle([u, d, r, l])
        elif not prime:
            cycle([u, r, d, l])
        else:
            cycle([u, l, d, r])

    def move_D(prime=False, half=False):
        rot_face(27, prime, half)
        botF = [18 + i for i in [6,7,8]]
        botR = [9 + i for i in [6,7,8]]
        botB = [45 + i for i in [6,7,8]]
        botL = [36 + i for i in [6,7,8]]
        if half:
            cycle([botF, botB, botR, botL]); cycle([botF, botB, botR, botL])
        elif not prime:
            cycle([botF, botL, botB, botR])
        else:
            cycle([botF, botR, botB, botL])

    def move_L(prime=False, half=False):
        rot_face(36, prime, half)
        u = [0 + i for i in [0,3,6]]
        f = [18 + i for i in [0,3,6]]
        d = [27 + i for i in [0,3,6]]
        b = [45 + i for i in [8,5,2]]
        if half:
            cycle([u, d, f, b]); cycle([u, d, f, b])
        elif not prime:
            cycle([u, b, d, f])
        else:
            cycle([u, f, d, b])

    def move_B(prime=False, half=False):
        rot_face(45, prime, half)
        u = [0 + i for i in [0,1,2]]
        l = [36 + i for i in [0,3,6]]
        d = [27 + i for i in [8,7,6]]
        r = [9 + i for i in [8,5,2]]
        if half:
            cycle([u, d, r, l]); cycle([u, d, r, l])
        elif not prime:
            cycle([u, l, d, r])
        else:
            cycle([u, r, d, l])

    if move[0] == "U":
        if move.endswith("2"):
            move_U(half=True)
        elif move.endswith("'"):
            move_U(prime=True)
        else:
            move_U()
    elif move[0] == "R":
        if move.endswith("2"):
            move_R(half=True)
        elif move.endswith("'"):
            move_R(prime=True)
        else:
            move_R()
    elif move[0] == "F":
        if move.endswith("2"):
            move_F(half=True)
        elif move.endswith("'"):
            move_F(prime=True)
        else:
            move_F()
    elif move[0] == "D":
        if move.endswith("2"):
            move_D(half=True)
        elif move.endswith("'"):
            move_D(prime=True)
        else:
            move_D()
    elif move[0] == "L":
        if move.endswith("2"):
            move_L(half=True)
        elif move.endswith("'"):
            move_L(prime=True)
        else:
            move_L()
    elif move[0] == "B":
        if move.endswith("2"):
            move_B(half=True)
        elif move.endswith("'"):
            move_B(prime=True)
        else:
            move_B()
    return s


# ---------- Search algorithms ----------


def bfs_solve(start: List[str], max_depth: int = 8):
    from collections import deque

    t0 = time.perf_counter()
    if is_goal(start):
        return [], 1, int((time.perf_counter() - t0) * 1000)

    frontier = deque([(start, [])])
    visited = {tuple(start)}
    explored = 0
    allowed_moves: List[Move] = ["U", "U'", "R", "R'", "F", "F'"]

    while frontier:
        state, path = frontier.popleft()
        explored += 1
        if len(path) >= max_depth:
            continue
        for mv in allowed_moves:
            nxt = apply_move(state, mv)
            key = tuple(nxt)
            if key in visited:
                continue
            visited.add(key)
            new_path = path + [mv]
            if is_goal(nxt):
                return new_path, explored, int((time.perf_counter() - t0) * 1000)
            frontier.append((nxt, new_path))

    return [], explored, int((time.perf_counter() - t0) * 1000)


def heuristic_lower_bound(state: List[str]) -> int:
    # Lightweight admissible heuristic combining orientation lower bounds and facelet mismatch
    # - corner orientation: a face turn twists 4 corners ⇒ ceil(#misoriented_corners / 4)
    # - edge orientation: F/B turns flip 4 edges ⇒ ceil(#flipped_edges / 4)
    co, eo = count_corner_edge_orientation(state)
    h_orient = max((co + 3) // 4, (eo + 3) // 4)
    # Very weak placement bound: stickers not matching their face center; one move can fix at most 8
    mismatch = 0
    for i in range(0, 54, 9):
        center = state[i + 4]
        face = state[i : i + 9]
        mismatch += sum(1 for c in face if c != center)
    h_place = (mismatch + 7) // 8
    return max(h_orient, h_place)


def count_corner_edge_orientation(state: List[str]) -> tuple[int, int]:
    # Corner (8) and Edge (12) facelet index tables (URFDLB ordering)
    C = [
        8, 9 + 0, 18 + 2,  # URF: U8 R0 F2
        6, 18 + 0, 36 + 2, # UFL: U6 F0 L2
        0, 36 + 0, 45 + 2, # ULB: U0 L0 B2
        2, 45 + 0, 9 + 2,  # UBR: U2 B0 R2
        27 + 2, 18 + 8, 9 + 6,   # DFR
        27 + 0, 36 + 8, 18 + 6,  # DLF
        27 + 6, 45 + 8, 36 + 6,  # DBL
        27 + 8, 9 + 8, 45 + 6,   # DRB
    ]
    E = [
        5, 9 + 1,      # UR
        7, 18 + 1,     # UF
        3, 36 + 1,     # UL
        1, 45 + 1,     # UB
        27 + 5, 9 + 7, # DR
        27 + 7, 18 + 7,# DF
        27 + 3, 36 + 7,# DL
        27 + 1, 45 + 7,# DB
        18 + 5, 9 + 3, # FR
        18 + 3, 36 + 5,# FL
        45 + 5, 36 + 3,# BL
        45 + 3, 9 + 5, # BR
    ]

    corner_names = [
        ("U", "R", "F"), ("U", "F", "L"), ("U", "L", "B"), ("U", "B", "R"),
        ("D", "F", "R"), ("D", "L", "F"), ("D", "B", "L"), ("D", "R", "B"),
    ]
    edge_names = [
        ("U", "R"), ("U", "F"), ("U", "L"), ("U", "B"),
        ("D", "R"), ("D", "F"), ("D", "L"), ("D", "B"),
        ("F", "R"), ("F", "L"), ("B", "L"), ("B", "R"),
    ]

    misoriented_corners = 0
    misoriented_edges = 0

    # Corners: count if U/D color is not on U/D facelet position among the three
    for i in range(8):
        cols = [state[C[3 * i + 0]], state[C[3 * i + 1]], state[C[3 * i + 2]]]
        # If this corner doesn't contain U or D, orientation determined by F/B vs L/R; approximate: don't count
        if not any(c in ("U", "D") for c in cols):
            continue
        pos_ud_first = cols[0] in ("U", "D")
        # Expect U/D to be at index 0 in our corner listing for top/bottom layers
        if not pos_ud_first:
            misoriented_corners += 1

    # Edges: if U/D edge, U/D sticker should be first; else F/B sticker should be first
    for i in range(12):
        c0, c1 = state[E[2 * i + 0]], state[E[2 * i + 1]]
        has_ud = (c0 in ("U", "D")) or (c1 in ("U", "D"))
        if has_ud:
            expected_ud_first = edge_names[i][0] in ("U", "D")
            actual_ud_first = c0 in ("U", "D")
            if expected_ud_first != actual_ud_first:
                misoriented_edges += 1
        else:
            expected_fb_first = edge_names[i][0] in ("F", "B")
            actual_fb_first = c0 in ("F", "B")
            if expected_fb_first != actual_fb_first:
                misoriented_edges += 1

    return misoriented_corners, misoriented_edges


def astar_solve(*_args, **_kwargs):
    # Disabled to adhere to project constraint (IDA* only)
    return [], 0, 0, None


def ida_star_solve(start: List[str], max_depth: int = 40):
    # Iterative Deepening A* using heuristic as cost bound with pruning and simple transposition table
    t0 = time.perf_counter()
    if is_goal(start):
        return [], 1, int((time.perf_counter() - t0) * 1000)

    allowed_moves: List[Move] = [
        "U","U'","U2","D","D'","D2","L","L'","L2","R","R'","R2","F","F'","F2","B","B'","B2"
    ]
    explored = 0

    best_g_for_state: Dict[tuple, int] = {}

    def search(path: List[str], state: List[str], g_cost: int, bound: int, last_move: str | None, last_face: str | None):
        nonlocal explored
        explored += 1
        h = heuristic_lower_bound(state)
        f = g_cost + h
        if f > bound:
            return None, f
        if is_goal(state):
            return path, f
        if g_cost >= max_depth:
            return None, float("inf")
        key = tuple(state)
        prev_best = best_g_for_state.get(key)
        if prev_best is not None and g_cost >= prev_best:
            return None, float("inf")
        best_g_for_state[key] = g_cost
        min_bound = float("inf")
        for mv in allowed_moves:
            # simple pruning: avoid immediate inverse
            if last_move and ((last_move == "U" and mv == "U'") or (last_move == "U'" and mv == "U") or
                               (last_move == "R" and mv == "R'") or (last_move == "R'" and mv == "R") or
                               (last_move == "F" and mv == "F'") or (last_move == "F'" and mv == "F")):
                continue
            # avoid repeating same face three times in a row
            face = mv[0]
            if last_face and face == last_face:
                if len(path) >= 1 and path[-1][0] == last_face:
                    continue
            nxt = apply_move(state, mv)
            res, tbound = search(path + [mv], nxt, g_cost + 1, bound, mv, face)
            if res is not None:
                return res, tbound
            if tbound < min_bound:
                min_bound = tbound
        return None, min_bound

    bound = heuristic_lower_bound(start)
    path: List[str] | None
    while True:
        best_g_for_state.clear()
        path, new_bound = search([], start, 0, bound, None, None)
        if path is not None:
            return path, explored, int((time.perf_counter() - t0) * 1000)
        if new_bound == float("inf"):
            return [], explored, int((time.perf_counter() - t0) * 1000)
        bound = int(new_bound)


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    state = req.state
    if len(state) != 54:
        raise ValueError("state must have 54 entries")

    start_is_goal = is_goal(state)
    moves, explored, time_ms = ida_star_solve(state)
    trace = None

    found = start_is_goal or len(moves) > 0
    return SolveResponse(
        moves=moves,
        explored_states=explored,
        time_ms=time_ms,
        depth=len(moves),
        meta={"algorithm": "ida*", "found": found},
        trace=trace,
    )


class ScrambleResponse(BaseModel):
    state: List[str]


@app.get("/scramble", response_model=ScrambleResponse)
def scramble() -> ScrambleResponse:
    # Generate a valid scramble by applying random legal moves from a subset
    import random
    state = []
    for face in FACE_ORDER:
        state.extend([face] * 9)
    allowed: List[Move] = ["U", "U'", "R", "R'", "F", "F'"]
    last = None
    for _ in range(12):
        candidates = [m for m in allowed if not (last and ((last == "U" and m == "U'") or (last == "U'" and m == "U") or (last == "R" and m == "R'") or (last == "R'" and m == "R") or (last == "F" and m == "F'") or (last == "F'" and m == "F")))]
        mv = random.choice(candidates)
        state = apply_move(state, mv)
        last = mv
    return ScrambleResponse(state=state)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Mount static last so API routes like POST /solve take precedence
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


