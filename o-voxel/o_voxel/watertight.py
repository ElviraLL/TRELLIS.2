"""
Watertight mesh extraction from Flexible Dual Grid (O-Voxel) data.

The standard ``flexible_dual_grid_to_mesh`` is sign-free: it emits one quad for
every intersected edge whose 4 surrounding voxels are active. This faithfully
reproduces open surfaces and non-manifold geometry, but it also reproduces
double-layer walls, enclosed inner geometry (mouth bags, teeth, inner clothing
layers) and spurious duplicated sheets, and its output is neither closed nor
consistently oriented.

This module extracts a *watertight, consistently oriented* outer surface
instead. It recovers an inside/outside sign field on the corner lattice by
flood-filling from outside the object, where the intersected flags (and
optionally edges buried inside the active-voxel shell) act as barriers. Faces
are then emitted only on lattice edges whose two endpoint corners carry
different signs, using the QEF dual vertices for geometry:

- Duplicated sheets around the same wall collapse to the outermost one.
- Geometry fully enclosed by the outer surface (inner shells, cavities'
  contents) is culled, and enclosed space becomes solid.
- Missing flags (pinholes) are sealed as long as the active-voxel shell
  itself is closed.
- Dangling open sheets that do not bound any volume are dropped.

The implementation is pure NumPy/SciPy and does not require the CUDA
extension; torch tensors (CPU or CUDA) are accepted and returned.
"""

from typing import *
import numpy as np

__all__ = [
    "flexible_dual_grid_to_watertight_mesh",
]

# For axis a, the other two axes (b, c) such that (a, b, c) is a right-handed
# cyclic permutation of (x, y, z).
_PERP = [(1, 2), (2, 0), (0, 1)]


def _to_numpy(x):
    if x is None or isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _shift_or8(v):
    """OR of the 8 voxels adjacent to each lattice corner.

    v: (Dx, Dy, Dz) bool voxel grid. Returns (Dx+1, Dy+1, Dz+1) bool corner
    grid where entry c is True iff any of the up-to-8 voxels touching corner c
    is True (out-of-range voxels count as False).
    """
    p = np.pad(v, 1)
    D = v.shape
    out = np.zeros((D[0] + 1, D[1] + 1, D[2] + 1), dtype=bool)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                out |= p[dx:dx + D[0] + 1, dy:dy + D[1] + 1, dz:dz + D[2] + 1]
    return out


def _edge_all4_active(A, axis):
    """For each lattice edge along `axis` at lower corner c, whether the 4
    voxels sharing that edge are all active.

    The edge c -> c + e_a crosses voxel slab ``v[axis] = c[axis]`` and is
    shared by voxels ``v[b] in {c[b]-1, c[b]}``, ``v[c] in {c[c]-1, c[c]}``.
    A: (Dx, Dy, Dz) bool. Returns a corner-grid-shaped bool array; entries
    whose edge leaves the grid are False (out-of-range voxels count inactive).
    """
    b, c = _PERP[axis]
    p = np.pad(A, 1)
    D = A.shape
    out = np.ones((D[0] + 1, D[1] + 1, D[2] + 1), dtype=bool)
    for db in range(2):
        for dc in range(2):
            sl = [None, None, None]
            sl[axis] = slice(1, D[axis] + 2)      # v[axis] = c[axis] (pad +1)
            sl[b] = slice(db, db + D[b] + 1)      # v[b] = c[b] - 1 + db
            sl[c] = slice(dc, dc + D[c] + 1)
            out &= p[tuple(sl)]
    return out


def flexible_dual_grid_to_watertight_mesh(
    coords,
    dual_vertices,
    intersected_flag,
    aabb: Union[list, tuple, np.ndarray],
    voxel_size: Union[float, list, tuple, np.ndarray] = None,
    grid_size: Union[int, list, tuple, np.ndarray] = None,
    mode: str = "solidify",
    seal_radius: int = 3,
    seal_active_edges: bool = True,
    max_iters: int = 256,
    verbose: bool = False,
):
    """
    Extract a watertight, consistently oriented mesh from flexible dual grid data.

    An inside/outside sign field is recovered on the corner lattice; a quad is
    then emitted for every lattice edge whose two endpoint corners carry
    different signs, connecting the dual vertices of the 4 voxels sharing the
    edge, wound so that face normals point outward.

    Two sign-recovery modes are available:

    - ``"solidify"`` (default, robust): the active voxel set is solidified by
      dilate(seal_radius) -> fill enclosed holes -> erode(seal_radius), which
      seals shell holes up to ~2*seal_radius voxels wide and fills the
      enclosed interior. A corner is inside iff its 8 adjacent voxels are all
      solid. Robust to the imperfect shells of generated data (open hair
      cards, truncated limbs, missing patches); open sheets become thin
      closed slabs instead of leaking.
    - ``"flood"`` (exact): flood fill from outside over the corner lattice,
      blocked by intersected edges (and, if `seal_active_edges`, edges whose
      4 surrounding voxels are all active). Places the surface exactly on the
      flagged edges, but any hole in the active shell lets the flood wash
      into the interior — use only for data whose shell is known to be
      closed (e.g. GT conversions of clean meshes).

    In both modes duplicated parallel sheets collapse to the outermost one,
    enclosed inner geometry is culled, and the enclosed interior becomes
    solid.

    Args:
        coords: (N, 3) int voxel coordinates of active voxels.
        dual_vertices: (N, 3) float dual vertex positions, local to each voxel
            (same convention as ``flexible_dual_grid_to_mesh``: the world
            position is ``(coords + dual_vertices) * voxel_size + aabb[0]``).
        intersected_flag: (N, 3) bool intersected flags (unused topologically
            in "solidify" mode; geometry always uses the dual vertices).
        aabb: (2, 3) axis-aligned bounding box.
        voxel_size / grid_size: one of the two must be provided.
        mode: sign recovery mode, "solidify" or "flood" (see above).
        seal_radius: hole-sealing radius in voxels for "solidify" mode.
        seal_active_edges: "flood" mode only — additionally block flood-fill
            on edges whose 4 surrounding voxels are all active, sealing
            pinholes (missing flags) wherever the active shell is closed.
        max_iters: "flood" mode only — safety cap on flood-fill sweeps.
        verbose: print statistics.

    Returns:
        vertices: (V, 3) float32 world-space vertices.
        faces: (F, 3) int64 triangles, counter-clockwise seen from outside.
        Returned as torch tensors if the inputs were torch tensors (on CPU),
        numpy arrays otherwise.
    """
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError("flexible_dual_grid_to_watertight_mesh requires scipy") from e

    return_torch = hasattr(coords, "detach")
    return_device = coords.device if return_torch else None

    coords = _to_numpy(coords).astype(np.int64)
    dual_vertices = _to_numpy(dual_vertices).astype(np.float32)
    intersected = _to_numpy(intersected_flag).astype(bool)
    aabb = _to_numpy(aabb).astype(np.float32)

    if voxel_size is not None:
        voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype=np.float32), (3,)).copy()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        grid_size = np.broadcast_to(np.asarray(grid_size, dtype=np.int64), (3,)).copy()
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    N = coords.shape[0]
    assert dual_vertices.shape == (N, 3)
    assert intersected.shape == (N, 3)

    # ------------------------------------------------------------------ #
    # Crop to the bounding box of active voxels with a 1-voxel empty pad #
    # ------------------------------------------------------------------ #
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    offset = mn - 1
    D = (mx - mn + 1) + 2  # voxel grid dims, empty ring guaranteed
    vc = coords - offset

    A = np.zeros(tuple(D), dtype=bool)
    A[vc[:, 0], vc[:, 1], vc[:, 2]] = True

    C = tuple(D + 1)
    if mode == "solidify":
        # ------------------------------------------------------ #
        # Solidify the active shell: dilate -> fill -> erode.    #
        # Seals holes up to ~2*seal_radius wide and fills the    #
        # enclosed interior, then sign corners against the solid #
        # ------------------------------------------------------ #
        solid = ndimage.binary_dilation(A, iterations=seal_radius)
        solid = ndimage.binary_fill_holes(solid)
        solid = ndimage.binary_erosion(solid, iterations=seal_radius, border_value=0)
        solid |= A
        # corner is outside iff any of its 8 adjacent voxels is non-solid
        S = _shift_or8(~solid)
        del solid
    elif mode == "flood":
        # ---------------------------------------------------------- #
        # Voxel-level empty space labeling: which empty voxels are   #
        # connected to the outside (6-connectivity)?                 #
        # ---------------------------------------------------------- #
        labels, _ = ndimage.label(~A)
        border_labels = np.unique(np.concatenate([
            labels[0].ravel(), labels[-1].ravel(),
            labels[:, 0].ravel(), labels[:, -1].ravel(),
            labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
        ]))
        border_labels = border_labels[border_labels != 0]
        outside_empty = np.isin(labels, border_labels)
        del labels

        # ------------------------------------------------------ #
        # Edge cuts on the corner lattice                        #
        # ------------------------------------------------------ #
        # Flag of voxel v on axis a refers to the lattice edge along a located
        # at v's max corner in the two perpendicular axes:
        #     lower corner c = v + e_b + e_c,   edge c -> c + e_a
        cuts = []
        for a in range(3):
            b, c = _PERP[a]
            cut = np.zeros(C, dtype=bool)
            holders = vc[intersected[:, a]]
            cc = holders.copy()
            cc[:, b] += 1
            cc[:, c] += 1
            cut[cc[:, 0], cc[:, 1], cc[:, 2]] = True
            if seal_active_edges:
                cut |= _edge_all4_active(A, a)
            cuts.append(cut)

        # ------------------------------------------------------ #
        # Flood fill outside signs over the corner lattice       #
        # ------------------------------------------------------ #
        S = _shift_or8(outside_empty)  # corners touching outside-connected empty space

        for it in range(max_iters):
            before = int(S.sum())
            for a in range(3):
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[a] = slice(0, C[a] - 1)   # corner c (edge c -> c + e_a)
                sl_hi[a] = slice(1, C[a])       # corner c + e_a
                sl_lo, sl_hi = tuple(sl_lo), tuple(sl_hi)
                open_edge = ~cuts[a][sl_lo]
                # monotone in-place ORs are safe under the overlapping views
                S[sl_hi] |= S[sl_lo] & open_edge
                S[sl_lo] |= S[sl_hi] & open_edge
            if int(S.sum()) == before:
                break
        else:
            import warnings
            warnings.warn(f"flood fill did not converge within {max_iters} sweeps")
        del cuts
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'solidify' or 'flood')")

    # ------------------------------------------------------ #
    # Resolve checkerboard plaquettes (pinch edges).         #
    # A 2x2 corner plaquette whose signs alternate           #
    # diagonally emits 4 faces sharing one mesh edge (non-   #
    # manifold). Flip one outside corner to inside (locally  #
    # growing the solid) until no such plaquette remains.    #
    # ------------------------------------------------------ #
    for _ in range(64):
        n_flip = 0
        for a in range(3):
            b, c = _PERP[a]
            sl00 = [slice(None)] * 3
            sl10 = [slice(None)] * 3
            sl01 = [slice(None)] * 3
            sl11 = [slice(None)] * 3
            sl00[b] = slice(0, C[b] - 1); sl00[c] = slice(0, C[c] - 1)
            sl10[b] = slice(1, C[b]);     sl10[c] = slice(0, C[c] - 1)
            sl01[b] = slice(0, C[b] - 1); sl01[c] = slice(1, C[c])
            sl11[b] = slice(1, C[b]);     sl11[c] = slice(1, C[c])
            s00, s10 = S[tuple(sl00)], S[tuple(sl10)]
            s01, s11 = S[tuple(sl01)], S[tuple(sl11)]
            # checkerboard: main diagonal equal, anti-diagonal equal, differ
            cb = (s00 == s11) & (s10 == s01) & (s00 != s10)
            if not cb.any():
                continue
            # flip an outside (True) corner to inside: s00 if s00 is the
            # outside diagonal, else s10
            flip00 = cb & s00
            flip10 = cb & s10
            v = S[tuple(sl00)]
            v &= ~flip00
            v = S[tuple(sl10)]
            v &= ~flip10
            n_flip += int(flip00.sum()) + int(flip10.sum())
        if n_flip == 0:
            break

    # ------------------------------------------------------ #
    # Emit quads on sign-change edges                        #
    # ------------------------------------------------------ #
    # Voxel id lookup: linear ids of active voxels, searchsorted for queries.
    def linear_id(v):
        return (v[:, 0] * D[1] + v[:, 1]) * D[2] + v[:, 2]

    active_ids = linear_id(vc)
    order = np.argsort(active_ids)
    active_ids_sorted = active_ids[order]

    quad_voxels = []      # (L, 4, 3) voxel coords (cropped)
    quad_outward_hi = []  # (L,) True if outside at corner c + e_a
    for a in range(3):
        b, c_ax = _PERP[a]
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[a] = slice(0, C[a] - 1)
        sl_hi[a] = slice(1, C[a])
        diff = S[tuple(sl_lo)] != S[tuple(sl_hi)]
        cs = np.argwhere(diff)  # lower corner c of each sign-change edge
        if cs.shape[0] == 0:
            continue
        # 4 voxels sharing edge c -> c+e_a, cyclic CCW in the (b, c_ax) plane:
        # (c[b]-1, c[c]-1) -> (c[b], c[c]-1) -> (c[b], c[c]) -> (c[b]-1, c[c])
        base = cs.copy()
        base[:, b] -= 1
        base[:, c_ax] -= 1
        q = np.repeat(base[:, None, :], 4, axis=1)
        q[:, 1, b] += 1
        q[:, 2, b] += 1
        q[:, 2, c_ax] += 1
        q[:, 3, c_ax] += 1
        quad_voxels.append(q)
        quad_outward_hi.append(S[tuple(sl_hi)][diff])
    if len(quad_voxels) == 0:
        empty_v = np.zeros((0, 3), dtype=np.float32)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        if return_torch:
            import torch
            return torch.from_numpy(empty_v).to(return_device), torch.from_numpy(empty_f).to(return_device)
        return empty_v, empty_f
    quad_voxels = np.concatenate(quad_voxels, axis=0)
    quad_outward_hi = np.concatenate(quad_outward_hi, axis=0)

    # CCW order in the (b, c) plane gives a +a normal; if the outside is on
    # the -a side (outside at lower corner c), reverse to flip the normal.
    flip = ~quad_outward_hi
    quad_voxels[flip] = quad_voxels[flip][:, ::-1]

    # ------------------------------------------------------ #
    # Vertices: dual vertex if the voxel is active,          #
    # voxel center otherwise                                 #
    # ------------------------------------------------------ #
    flat = quad_voxels.reshape(-1, 3)
    flat_ids = linear_id(flat)
    pos = np.searchsorted(active_ids_sorted, flat_ids)
    pos_c = np.clip(pos, 0, len(active_ids_sorted) - 1)
    found = active_ids_sorted[pos_c] == flat_ids
    vert_idx = np.empty(len(flat_ids), dtype=np.int64)
    vert_idx[found] = order[pos_c[found]]

    missing_ids = flat_ids[~found]
    uniq_missing, inv = np.unique(missing_ids, return_inverse=True)
    vert_idx[~found] = N + inv

    vertices = np.empty((N + len(uniq_missing), 3), dtype=np.float32)
    vertices[:N] = ((coords + dual_vertices) * voxel_size + aabb[0]).astype(np.float32)
    if len(uniq_missing) > 0:
        mz = uniq_missing % D[2]
        my = (uniq_missing // D[2]) % D[1]
        mxs = uniq_missing // (D[1] * D[2])
        mcoords = np.stack([mxs, my, mz], axis=-1) + offset
        vertices[N:] = ((mcoords + 0.5) * voxel_size + aabb[0]).astype(np.float32)

    quads = vert_idx.reshape(-1, 4)

    # ------------------------------------------------------ #
    # Triangulate: pick the diagonal whose two triangle      #
    # normals agree the most (same heuristic as the sign-    #
    # free extractor, but orientation-preserving)            #
    # ------------------------------------------------------ #
    p0, p1, p2, p3 = (vertices[quads[:, i]] for i in range(4))
    n0a = np.cross(p1 - p0, p2 - p0)
    n0b = np.cross(p2 - p0, p3 - p0)
    align0 = (n0a * n0b).sum(axis=1)
    n1a = np.cross(p1 - p0, p3 - p0)
    n1b = np.cross(p2 - p1, p3 - p1)
    align1 = (n1a * n1b).sum(axis=1)
    use0 = align0 >= align1
    tris = np.empty((quads.shape[0], 2, 3), dtype=np.int64)
    tris[use0, 0] = quads[use0][:, [0, 1, 2]]
    tris[use0, 1] = quads[use0][:, [0, 2, 3]]
    tris[~use0, 0] = quads[~use0][:, [0, 1, 3]]
    tris[~use0, 1] = quads[~use0][:, [1, 2, 3]]
    faces = tris.reshape(-1, 3)

    # Drop vertices not referenced by any face
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    vertices = vertices[used]
    faces = remap[faces]

    if verbose:
        n_out = int(S.sum())
        print(f"[watertight] corners: {S.size} ({n_out} outside), "
              f"quads: {len(quads)}, faces: {len(faces)}, vertices: {len(vertices)}")

    if return_torch:
        import torch
        return (
            torch.from_numpy(np.ascontiguousarray(vertices)).to(return_device),
            torch.from_numpy(np.ascontiguousarray(faces)).to(return_device),
        )
    return vertices, faces
