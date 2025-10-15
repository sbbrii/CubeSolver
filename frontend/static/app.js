// Minimal 3D cube with selectable stickers and simple move playback.
// For academic clarity, state is 54-length array over faces U,R,F,D,L,B.

const FACE_ORDER = ["U", "R", "F", "D", "L", "B"];
const LEGAL_MOVES = [
  "U","U'","U2","D","D'","D2","L","L'","L2","R","R'","R2","F","F'","F2","B","B'","B2"
];
const FACE_COLORS = {
  U: 0xffffff,
  R: 0xff6b6b,
  F: 0x00d084,
  D: 0xffd166,
  L: 0x4dabf7,
  B: 0x9b5de5,
};

let state = FACE_ORDER.flatMap((f) => Array(9).fill(f));
let suppressUI = false;

const canvas = document.getElementById("scene");
const container = document.querySelector(".stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

function resizeRenderer() {
  const rect = container.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f1a);

const camera = new THREE.PerspectiveCamera(45, 1.6, 0.1, 100);
camera.position.set(4, 4, 6);
resizeRenderer();

// Manual drag controls
let isDragging = false;
let lastX = 0, lastY = 0;
renderer.domElement.addEventListener("pointerdown", (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("pointerup", () => { isDragging = false; });
window.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  const rotY = dx * 0.005;
  const rotX = dy * 0.005;
  const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0), rotX);
  const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0), rotY);
  group.applyQuaternion(qx.multiply(qy));
});

const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
scene.add(light);

const group = new THREE.Group();
scene.add(group);

// Build 3x3 faces as quads we can click
const faceMeshes = []; // 54 meshes
const size = 1;
const gap = 0.02;

function createFace(faceIndex, normal, up, right, colorKey) {
  const base = new THREE.Vector3().copy(normal).multiplyScalar(1.5 * size);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const idx = faceIndex * 9 + r * 3 + c;
      const center = new THREE.Vector3()
        .copy(base)
        .addScaledVector(up, (1 - r) * size)
        .addScaledVector(right, (c - 1) * size);
      const geom = new THREE.PlaneGeometry(1 - gap, 1 - gap);
      const mat = new THREE.MeshBasicMaterial({ color: FACE_COLORS[colorKey] });
      const m = new THREE.Mesh(geom, mat);
      // orient plane to face normal
      const look = new THREE.Vector3().addVectors(center, normal);
      m.position.copy(center);
      m.lookAt(look);
      m.userData = { faceIndex, localIndex: r * 3 + c };
      group.add(m);
      faceMeshes.push(m);
    }
  }
}

createFace(0, new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(1, 0, 0), "U");
createFace(1, new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, -1), "R");
createFace(2, new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0), "F");
createFace(3, new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0), "D");
createFace(4, new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), "L");
createFace(5, new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1, 0), new THREE.Vector3(-1, 0, 0), "B");

// Center the cube and frame it for the camera
function centerAndFrame() {
  const box = new THREE.Box3().setFromObject(group);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  // move group so its center is at the origin
  group.position.x -= center.x;
  group.position.y -= center.y;
  group.position.z -= center.z;
  // position camera at a distance so object fits
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = (camera.fov * Math.PI) / 180;
  const dist = (maxDim / 2) / Math.tan(fov / 2) + 2; // margin
  camera.position.set(dist, dist, dist);
  camera.lookAt(0, 0, 0);
}

centerAndFrame();

function refreshColors() {
  for (let i = 0; i < 54; i++) {
    const key = state[i];
    faceMeshes[i].material.color.setHex(FACE_COLORS[key]);
  }
}

// ---- Legality checks ----
function isLegalMoveNotation(mv) {
  return LEGAL_MOVES.includes(mv);
}

function isLegalCubeState(st) {
  if (!Array.isArray(st) || st.length !== 54) return { ok: false, reason: "State must have 54 stickers" };
  const counts = { U:0, R:0, F:0, D:0, L:0, B:0 };
  for (const c of st) {
    if (!FACE_ORDER.includes(c)) return { ok: false, reason: `Invalid color '${c}'` };
    counts[c]++;
  }
  for (const face of FACE_ORDER) {
    if (counts[face] !== 9) return { ok: false, reason: `Face ${face} must have exactly 9 stickers` };
  }
  // Basic center consistency: centers define face colors
  const centers = [st[4], st[13], st[22], st[31], st[40], st[49]];
  for (const c of centers) {
    if (!FACE_ORDER.includes(c)) return { ok: false, reason: "Invalid center color" };
  }
  // Deep physical invariants (corner/edge orientation sums and permutation parity)
  const inv = checkCubieInvariants(st);
  if (!inv.ok) return inv;
  return { ok: true };
}

// ---- Cubie invariants (orientation sums and permutation parity) ----
function checkCubieInvariants(st) {
  // Corner and edge facelet index tables in URFDLB face order
  const C = [
    [ 0+8, 9+0, 18+2], // URF: U8 R0 F2
    [ 0+6, 18+0, 36+2], // UFL: U6 F0 L2
    [ 0+0, 36+0, 45+2], // ULB: U0 L0 B2
    [ 0+2, 45+0, 9+2],  // UBR: U2 B0 R2
    [27+2, 18+8, 9+6],  // DFR: D2 F8 R6
    [27+0, 36+8, 18+6], // DLF: D0 L8 F6
    [27+6, 45+8, 36+6], // DBL: D6 B8 L6
    [27+8, 9+8, 45+6],  // DRB: D8 R8 B6
  ];
  const E = [
    [0+5, 9+1],  // UR
    [0+7, 18+1], // UF
    [0+3, 36+1], // UL
    [0+1, 45+1], // UB
    [27+5, 9+7], // DR
    [27+7, 18+7],// DF
    [27+3, 36+7],// DL
    [27+1, 45+7],// DB
    [18+5, 9+3], // FR
    [18+3, 36+5],// FL
    [45+5, 36+3],// BL
    [45+3, 9+5], // BR
  ];
  const cornerNames = [["U","R","F"],["U","F","L"],["U","L","B"],["U","B","R"],["D","F","R"],["D","L","F"],["D","B","L"],["D","R","B"]];
  const edgeNames = [["U","R"],["U","F"],["U","L"],["U","B"],["D","R"],["D","F"],["D","L"],["D","B"],["F","R"],["F","L"],["B","L"],["B","R"]];

  // Build permutation and orientation arrays
  let cornerPerm = new Array(8); let cornerOriSum = 0;
  let edgePerm = new Array(12); let edgeOriSum = 0;

  // Corner permutation and orientation
  for (let i=0;i<8;i++) {
    const idxs = C[i];
    const cols = [st[idxs[0]], st[idxs[1]], st[idxs[2]]];
    // Identify cubie by set of colors
    const targetIndex = cornerNames.findIndex(names => names.every(n => cols.includes(n)));
    if (targetIndex === -1) return { ok: false, reason: "Corner color set invalid" };
    cornerPerm[i] = targetIndex;
    // Determine orientation: position of U/D color within this corner's colors vs expected index 0
    const udColor = cols.find(c => c === "U" || c === "D");
    const pos = cols.indexOf(udColor);
    // In our table, for top/bottom corners, the U/D facelet is at position 0
    const ori = (pos === -1) ? 0 : (pos % 3);
    cornerOriSum = (cornerOriSum + ori) % 3;
  }

  // Edge permutation and orientation
  for (let i=0;i<12;i++) {
    const idxs = E[i];
    const cols = [st[idxs[0]], st[idxs[1]]];
    const targetIndex = edgeNames.findIndex(names => names.every(n => cols.includes(n)));
    if (targetIndex === -1) return { ok: false, reason: "Edge color set invalid" };
    edgePerm[i] = targetIndex;
    // Orientation: 0 if U/D color on U/D face OR if no U/D then F/B color on F/B face (approx)
    const hasUD = cols.includes("U") || cols.includes("D");
    let ori = 0;
    if (hasUD) {
      // Expect U/D sticker to be on a U/D facelet index (which is first face in edgeNames[i])
      const expectedUD = edgeNames[i][0] === "U" || edgeNames[i][0] === "D";
      const actualUDAtFirst = (cols[0] === "U" || cols[0] === "D");
      ori = (expectedUD === actualUDAtFirst) ? 0 : 1;
    } else {
      const firstIsFB = edgeNames[i][0] === "F" || edgeNames[i][0] === "B";
      const actualFBAtFirst = (cols[0] === "F" || cols[0] === "B");
      ori = (firstIsFB === actualFBAtFirst) ? 0 : 1;
    }
    edgeOriSum = (edgeOriSum + ori) % 2;
  }

  if (cornerOriSum !== 0) return { ok: false, reason: "Corner orientation sum must be 0 mod 3" };
  if (edgeOriSum !== 0) return { ok: false, reason: "Edge orientation sum must be 0 mod 2" };

  // Permutation parity: corners and edges must have same parity
  function parityOfPermutation(perm) {
    let seen = new Array(perm.length).fill(false), parity = 0;
    for (let i=0;i<perm.length;i++) {
      if (seen[i]) continue;
      let j=i, cycleLen=0;
      while(!seen[j]) { seen[j]=true; j=perm[j]; cycleLen++; }
      if (cycleLen>0) parity ^= ((cycleLen-1)&1);
    }
    return parity;
  }
  const cp = parityOfPermutation(cornerPerm);
  const ep = parityOfPermutation(edgePerm);
  if ((cp ^ ep) !== 0) return { ok: false, reason: "Corner/edge permutation parity mismatch" };
  return { ok: true };
}

// Simulate a move and check resulting state validity
function simulateMove(st, mv) {
  const copy = st.slice();
  // Reuse the local move functions on a copy by temporarily swapping global state
  const original = state;
  const originalSuppress = suppressUI;
  try {
    state = copy;
    suppressUI = true;
    applyLocalMove(mv);
    const out = state.slice();
    state = original;
    suppressUI = originalSuppress;
    return out;
  } catch (e) {
    state = original;
    suppressUI = originalSuppress;
    return null;
  }
}

function checkMoveLegality(cubeState, move) {
  if (!isLegalMoveNotation(move)) return false;
  const next = simulateMove(cubeState, move);
  if (!next) return false;
  const ok = isLegalCubeState(next);
  return !!ok.ok;
}

function updateMoveButtonsLegality() {
  const mapping = {
    "mv-U":"U","mv-Up":"U'","mv-U2":"U2",
    "mv-R":"R","mv-Rp":"R'","mv-R2":"R2",
    "mv-F":"F","mv-Fp":"F'","mv-F2":"F2",
    "mv-D":"D","mv-Dp":"D'","mv-D2":"D2",
    "mv-L":"L","mv-Lp":"L'","mv-L2":"L2",
    "mv-B":"B","mv-Bp":"B'","mv-B2":"B2"
  };
  const st = state.slice();
  for (const id in mapping) {
    const btn = document.getElementById(id);
    if (!btn) continue;
    const mv = mapping[id];
    btn.disabled = !checkMoveLegality(st, mv);
    btn.title = btn.disabled ? "Illegal Move" : "";
  }
}

// (Removed advanced local scramble in favor of backend-provided scramble)

refreshColors();

// Picking
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function cycleColor(faceletIndex) {
  // Prevent changing centers (indices 4,13,22,31,40,49)
  if (faceletIndex === 4 || faceletIndex === 13 || faceletIndex === 22 || faceletIndex === 31 || faceletIndex === 40 || faceletIndex === 49) {
    return;
  }
  const cur = state[faceletIndex];
  const order = FACE_ORDER;
  const next = order[(order.indexOf(cur) + 1) % order.length];
  state[faceletIndex] = next;
  refreshColors();
  onColorChanged();
}

renderer.domElement.addEventListener("pointerdown", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(faceMeshes);
  if (hits.length > 0) {
    const mesh = hits[0].object;
    const faceIndex = mesh.userData.faceIndex;
    const local = mesh.userData.localIndex;
    const idx = faceIndex * 9 + local;
    cycleColor(idx);
  }
});

function animate() {
  resizeRenderer();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

// Buttons
document.getElementById("btn-reset").onclick = () => {
  state = FACE_ORDER.flatMap((f) => Array(9).fill(f));
  refreshColors();
  updateMoveButtonsLegality();
  renderMetrics({ status: "ready", algorithm: "ida*" });
  const btnSolve = document.getElementById("btn-solve");
  if (btnSolve) btnSolve.disabled = false;
};

document.getElementById("btn-randomize").onclick = async () => {
  try {
    const res = await fetch("http://localhost:8000/scramble");
    const data = await res.json();
    state = data.state;
    refreshColors();
    updateMoveButtonsLegality();
    onColorChanged();
  } catch (e) {
    console.error(e);
  }
};

function renderMetrics(obj) {
  const el = document.getElementById("metrics");
  if (!obj) { el.textContent = ""; return; }
  if (obj.error) {
    el.innerHTML = `<div><strong>Status:</strong> error</div><div>${obj.error}</div>`;
    return;
  }
  if (obj.status) {
    el.innerHTML = `<div><strong>Status:</strong> ${obj.status}</div><div><strong>Algorithm:</strong> ${obj.algorithm || "-"}</div>`;
    return;
  }
  const lines = [
    `<div><strong>Algorithm:</strong> ${obj.meta?.algorithm || "-"}</div>`,
    `<div><strong>Depth:</strong> ${obj.depth ?? "-"}</div>`,
    `<div><strong>Explored:</strong> ${obj.explored_states ?? "-"}</div>`,
    `<div><strong>Time:</strong> ${obj.time_ms ?? "-"} ms</div>`,
    `<div><strong>Moves:</strong> ${((obj.moves || []).length ? obj.moves.join(" ") : "(none)")}</div>`,
    `${obj.meta && obj.meta.found === false ? `<div style="color:#f59e0b">Note: solution not found within limits</div>` : ""}`,
  ];
  el.innerHTML = lines.join("");
}

// ---- Validation helpers and control gating ----
function getValidationError(st) {
  const res = isLegalCubeState(st);
  if (!res.ok) return res.reason || "Invalid state";
  return null;
}

function validateCubeState(cubeState) {
  return getValidationError(cubeState) === null;
}

function setMoveButtonsDisabled(disabled) {
  const ids = [
    "mv-U","mv-Up","mv-U2","mv-R","mv-Rp","mv-R2","mv-F","mv-Fp","mv-F2",
    "mv-D","mv-Dp","mv-D2","mv-L","mv-Lp","mv-L2","mv-B","mv-Bp","mv-B2"
  ];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled ? true : el.disabled;
  }
}

function onColorChanged() {
  updateMoveButtonsLegality();
  const err = getValidationError(state);
  const btnSolve = document.getElementById("btn-solve");
  if (err) {
    renderMetrics({ error: `❌ Invalid Cube Configuration: ${err}` });
    if (btnSolve) btnSolve.disabled = true;
    setMoveButtonsDisabled(true);
  } else {
    if (btnSolve) btnSolve.disabled = false;
  }
}

async function solve() {
  const btnSolve = document.getElementById("btn-solve");
  const btnReset = document.getElementById("btn-reset");
  const btnRand = document.getElementById("btn-randomize");
  // Validate before attempting to solve
  const err = getValidationError(state);
  if (err) {
    renderMetrics({ error: `❌ Invalid Cube Configuration: ${err}` });
    if (btnSolve) btnSolve.disabled = true;
    setMoveButtonsDisabled(true);
    return;
  }
  btnSolve.disabled = true; btnReset.disabled = true; btnRand.disabled = true;
  renderMetrics({ status: "solving...", algorithm: "ida*" });
  const body = { state };
  try {
    const res = await fetch("http://localhost:8000/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      renderMetrics({ error: `HTTP ${res.status}: ${txt}` });
    } else {
      const data = await res.json();
      renderMetrics(data);
      await playbackMoves(data.moves);
    }
  } catch (e) {
    renderMetrics({ error: String(e) });
  }
  btnSolve.disabled = false; btnReset.disabled = false; btnRand.disabled = false;
}

document.getElementById("btn-solve").onclick = solve;

// Simple visual playback: rotate group briefly to hint movement (not accurate kinematics)
function rotateGroup(axis, angle) {
  const q = new THREE.Quaternion();
  q.setFromAxisAngle(axis, angle);
  group.applyQuaternion(q);
}

async function playbackMoves(moves) {
  for (const mv of moves) {
    const affected = affectedFacelets(mv);
    highlightFacelets(affected, true);
    applyLocalMove(mv); // keep state/colors in sync with backend nomenclature
    await new Promise((r) => setTimeout(r, 300));
    highlightFacelets(affected, false);
    await new Promise((r) => setTimeout(r, 80));
  }
}

function affectedFacelets(move) {
  const f = move[0];
  if (f === "U") return Array.from({ length: 9 }, (_, i) => 0 + i);
  if (f === "R") return Array.from({ length: 9 }, (_, i) => 9 + i);
  if (f === "F") return Array.from({ length: 9 }, (_, i) => 18 + i);
  if (f === "D") return Array.from({ length: 9 }, (_, i) => 27 + i);
  if (f === "L") return Array.from({ length: 9 }, (_, i) => 36 + i);
  if (f === "B") return Array.from({ length: 9 }, (_, i) => 45 + i);
  return [];
}

function highlightFacelets(indices, on) {
  for (const idx of indices) {
    const m = faceMeshes[idx].material;
    if (on) {
      m.color.offsetHSL(0, 0, 0.2);
    } else {
      // recompute from state to restore exact color
      const key = state[idx];
      m.color.setHex(FACE_COLORS[key]);
    }
  }
}

// ---- Local move application on facelets (U, R, F, D, L, B and variants) ----
function rotateFaceIndices(faceStart, prime=false, half=false) {
  const idx = Array.from({length:9}, (_,i)=>faceStart+i);
  const copy = state.slice();
  const cw = {0:6,1:3,2:0,3:7,4:4,5:1,6:8,7:5,8:2};
  const ccw = {0:2,1:5,2:8,3:1,4:4,5:7,6:0,7:3,8:6};
  const halfm = {0:8,1:7,2:6,3:5,4:4,5:3,6:2,7:1,8:0};
  const map = half ? halfm : (prime ? ccw : cw);
  for (let i=0;i<9;i++) state[idx[i]] = copy[idx[map[i]]];
}

function cycleStrips(strips) {
  const copy = state.slice();
  for (let i=0;i<4;i++) {
    const src = strips[(i+3)%4];
    const dst = strips[i];
    for (let j=0;j<3;j++) state[dst[j]] = copy[src[j]];
  }
}

function moveU(prime=false, half=false) {
  rotateFaceIndices(0, prime, half);
  const topF = [18,19,20], topR=[9,10,11], topB=[45,46,47], topL=[36,37,38];
  if (half) { cycleStrips([topF, topB, topR, topL]); cycleStrips([topF, topB, topR, topL]); }
  else if (!prime) cycleStrips([topF, topR, topB, topL]);
  else cycleStrips([topF, topL, topB, topR]);
}
function moveR(prime=false, half=false) {
  rotateFaceIndices(9, prime, half);
  const u=[2,5,8], f=[20,23,26], d=[29,32,35], b=[51,48,45];
  if (half) { cycleStrips([u,d,f,b]); cycleStrips([u,d,f,b]); }
  else if (!prime) cycleStrips([u,f,d,b]);
  else cycleStrips([u,b,d,f]);
}
function moveF(prime=false, half=false) {
  rotateFaceIndices(18, prime, half);
  const u=[6,7,8], r=[9,12,15], d=[27+2,27+1,27+0], l=[36+8,36+5,36+2];
  if (half) { cycleStrips([u,d,r,l]); cycleStrips([u,d,r,l]); }
  else if (!prime) cycleStrips([u,r,d,l]);
  else cycleStrips([u,l,d,r]);
}

function moveD(prime=false, half=false) {
  rotateFaceIndices(27, prime, half);
  const f=[24,25,26], r=[15,16,17], b=[51,52,53], l=[42,43,44];
  if (half) { cycleStrips([f,b,r,l]); cycleStrips([f,b,r,l]); }
  else if (!prime) cycleStrips([f,l,b,r]);
  else cycleStrips([f,r,b,l]);
}

function moveL(prime=false, half=false) {
  rotateFaceIndices(36, prime, half);
  const u=[0,3,6], f=[18,21,24], d=[27,30,33], b=[53,50,47];
  if (half) { cycleStrips([u,d,f,b]); cycleStrips([u,d,f,b]); }
  else if (!prime) cycleStrips([u,b,d,f]);
  else cycleStrips([u,f,d,b]);
}

function moveB(prime=false, half=false) {
  rotateFaceIndices(45, prime, half);
  const u=[0,1,2], l=[36,39,42], d=[33,34,35], r=[11,14,17];
  if (half) { cycleStrips([u,d,r,l]); cycleStrips([u,d,r,l]); }
  else if (!prime) cycleStrips([u,l,d,r]);
  else cycleStrips([u,r,d,l]);
}

function applyLocalMove(mv) {
  const base = mv[0], mod = mv.length>1 ? mv.slice(1) : "";
  const prime = mod==="'"; const half = mod==="2";
  if (base==="U") moveU(prime, half);
  if (base==="R") moveR(prime, half);
  if (base==="F") moveF(prime, half);
  if (base==="D") moveD(prime, half);
  if (base==="L") moveL(prime, half);
  if (base==="B") moveB(prime, half);
  if (!suppressUI) {
    refreshColors();
    updateMoveButtonsLegality();
  }
}

// Manual move buttons
function guardAndApply(mv) {
  if (!isLegalMoveNotation(mv)) { renderMetrics({ error: `Illegal move: ${mv}` }); return; }
  const check = isLegalCubeState(state);
  if (!check.ok) { renderMetrics({ error: `Illegal cube state: ${check.reason}` }); return; }
  applyLocalMove(mv);
}

document.getElementById("mv-U").onclick = () => guardAndApply("U");
document.getElementById("mv-Up").onclick = () => guardAndApply("U'");
document.getElementById("mv-U2").onclick = () => guardAndApply("U2");
document.getElementById("mv-R").onclick = () => guardAndApply("R");
document.getElementById("mv-Rp").onclick = () => guardAndApply("R'");
document.getElementById("mv-R2").onclick = () => guardAndApply("R2");
document.getElementById("mv-F").onclick = () => guardAndApply("F");
document.getElementById("mv-Fp").onclick = () => guardAndApply("F'");
document.getElementById("mv-F2").onclick = () => guardAndApply("F2");
document.getElementById("mv-D").onclick = () => guardAndApply("D");
document.getElementById("mv-Dp").onclick = () => guardAndApply("D'");
document.getElementById("mv-D2").onclick = () => guardAndApply("D2");
document.getElementById("mv-L").onclick = () => guardAndApply("L");
document.getElementById("mv-Lp").onclick = () => guardAndApply("L'");
document.getElementById("mv-L2").onclick = () => guardAndApply("L2");
document.getElementById("mv-B").onclick = () => guardAndApply("B");
document.getElementById("mv-Bp").onclick = () => guardAndApply("B'");
document.getElementById("mv-B2").onclick = () => guardAndApply("B2");

// Initial legality update on load
updateMoveButtonsLegality();


