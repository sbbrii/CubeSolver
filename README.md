## Rubik's Cube Solver (CST401 KTU)

This project is a full‑stack AI Rubik's Cube solver designed for the CST401 (Artificial Intelligence) course at KTU. It demonstrates state‑space representation, goal testing, search strategies (BFS, A*, IDA*), and performance metrics.

### Tech Stack
- Backend: FastAPI (Python) for solver algorithms and REST API
- Frontend: React + Three.js (react‑three‑fiber) for 3D cube visualization

### Run Backend
```bash
cd backend
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Endpoints:
- `GET /health` – health check
- `GET /scramble` – sample scrambled state (54 facelets)
- `POST /solve` – body: `{ state: string[54], algorithm: "bfs"|"astar"|"ida*" }`

### Frontend
If Node.js 20.19+ is available:
```bash
cd frontend
npm create vite@latest rubiks-ui -- --template react-ts
cd rubiks-ui
npm install three @react-three/fiber @react-three/drei axios zustand tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm run dev
```

If your Node version is below Vite's requirement, either upgrade Node or use an alternative like CRA or a prebuilt static HTML + CDN Three.js setup.

### AI Concepts Mapping
- State space: 54‑length facelet vector in `backend/main.py` (`state: List[str]`)
- Initial state: user‑provided or `/scramble`
- Goal test: `is_goal(state)` checks uniform colors per face
- Actions: quarter turns (U, R, F and inverses); `apply_move`
- Transition model: `apply_move` (currently a placeholder for index permutations)
- Search strategies: `bfs_solve`, `astar_solve`, `ida_star_solve`
- Heuristic: `heuristic_misplaced_faces` (facelet mismatch count)
- Metrics: explored states, depth, elapsed time (ms)

### Next Steps
- Implement full cube move permutations for all faces (U, D, L, R, F, B and 2/prime variants)
- Wire the interactive 3D cube to produce/consume the 54‑facelet state
- Visualize the search process and step‑by‑step solution playback

### Academic Notes
Code comments highlight AI components for evaluation. The heuristic here is simple and intended for demonstration; you can replace it with domain heuristics (e.g., pattern databases, corner/edge orientation counts) and compare A* vs IDA* empirically using the provided metrics.


