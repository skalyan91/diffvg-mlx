"""
    Batched BVH construction on the GPU (MLX) for the Metal backend.

    Builds every BVH of one kind (all path BVHs, all group BVHs, or the scene
    BVH) in a single batched computation, producing exactly the node layout of
    scene.cpp `build_bvh` (see pydiffvg/metal/common.metal BVH accessors):

    - per BVH, leaves are ordered by sort key ascending and stored at local
      indices 0..n-1;
    - internal nodes are appended level by level with the same
      prev_beg / prev_end / leftover pairing as build_bvh, so the root is local
      index 2n-2 and child indices are LOCAL to the BVH;
    - internal box = merge(child0, child1), max_radius = max(child0, child1),
      with the tie behaviour of the C++ code (diffvg.h min/max return the
      second argument on ties, std::max the first), so signed zeros match too.

    Algorithm
    - Topology (host, numpy): the pairing pattern of build_bvh depends only on
      n. The C++ while-loop is simulated for all BVHs at once, vectorised over
      the BVHs (one numpy pass per loop iteration, ~log2(max n) + 2
      iterations). Iteration t of every BVH only reads nodes created in
      iterations < t of the same BVH, so all parents of iteration t (over all
      BVHs) are evaluated together. Nodes live in a "work order" buffer: the
      sorted leaves of all BVHs, then the parents of iteration 1, 2, ... as
      contiguous blocks; one final gather maps work order to the pool layout
      (BVH b occupies [base_b, base_b + 2 n_b - 1)). Plans are cached by
      num_leaves_per_bvh.
    - Leaf order (GPU): one argsort of a composite int64 key
      (bvh_id << 32) | key32, where key32 is an order-preserving uint32 image
      of the sort key (float32: IEEE bit trick; integers: as is). Ties get an
      arbitrary deterministic order (as allowed; std::sort is unstable).
    - Boxes (GPU): per iteration one gather of both children from the work
      buffer, elementwise min/max, and a contiguous slice update.
"""
import numpy as np
import mlx.core as mx

__all__ = ['BVHBuild', 'build_bvhs', 'refit_bvhs', 'morton2d_keys', 'build_plan',
           'clear_plan_cache']


class BVHBuild:
    """
        Result of build_bvhs.
        nodes_f  mx.float32 (N_total, 5): p_min.x, p_min.y, p_max.x, p_max.y, max_radius
        nodes_i  mx.int32   (N_total, 2): leaves: payload (sorted order); internal: local children
        base     np.int32   (B,): first node of each BVH (sum over earlier BVHs of 2 n - 1)
        root     np.int32   (B,): base + 2 n - 2
        perm     mx.int32   (L,): original leaf index of the k-th sorted leaf (work order);
                 leaves of BVH b are perm[leaf_start_b : leaf_start_b + n_b]
        plan     the topology plan (see build_plan)
    """
    __slots__ = ('nodes_f', 'nodes_i', 'base', 'root', 'perm', 'plan')

    def __init__(self, nodes_f, nodes_i, base, root, perm, plan):
        self.nodes_f = nodes_f
        self.nodes_i = nodes_i
        self.base = base
        self.root = root
        self.perm = perm
        self.plan = plan


class _Plan:
    __slots__ = ('num_leaves', 'base', 'leaf_start', 'num_nodes', 'num_leaves_total',
                 'levels', 'final_to_work', 'internal_children', 'mx_levels',
                 'mx_final_to_work', 'mx_internal_children', 'mx_leaf_bvh')


_PLAN_CACHE = {}
_PLAN_CACHE_MAX = 16


def clear_plan_cache():
    _PLAN_CACHE.clear()


def _ragged_arange(counts):
    """ concat(arange(c) for c in counts) and the owner index of each entry """
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    owner = np.repeat(np.arange(counts.shape[0], dtype = np.int64), counts)
    starts = np.cumsum(counts) - counts
    local = np.arange(total, dtype = np.int64) - np.repeat(starts, counts)
    return local, owner


def build_plan(num_leaves_per_bvh):
    """
        Host topology plan for BVHs with the given leaf counts (numpy int, all
        >= 1). Cached by the leaf counts. Contains
        - base (B,), leaf_start (B,), num_nodes, num_leaves_total
        - levels: list of (work_c0, work_c1) int32 arrays; the parents of
          level t occupy work positions [L + sum of earlier level sizes, ...)
        - final_to_work (N_total,): work position of every pool node
        - internal_children (N_total - L, 2) int32: LOCAL children of the
          internal nodes in work order
    """
    nl = np.ascontiguousarray(np.asarray(num_leaves_per_bvh).reshape(-1), dtype = np.int64)
    key = nl.tobytes()
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if nl.size > 0 and nl.min() < 1:
        raise ValueError('every BVH needs at least one leaf')
    B = nl.shape[0]
    L = int(nl.sum())
    sizes = 2 * nl - 1
    base = np.cumsum(sizes) - sizes
    leaf_start = np.cumsum(nl) - nl
    N = int(sizes.sum())

    final_to_work = np.full(N, -1, dtype = np.int64)
    leaf_local, leaf_owner = _ragged_arange(nl)
    final_to_work[base[leaf_owner] + leaf_local] = leaf_start[leaf_owner] + leaf_local

    # Vectorised simulation of the build_bvh while loop over the BVHs with n > 1.
    ids = np.nonzero(nl > 1)[0]
    prev_beg = np.zeros(ids.shape[0], dtype = np.int64)
    prev_end = nl[ids].copy()
    leftover = np.where(prev_end % 2 == 0, -1, prev_end - 1)
    levels = []
    children = []
    next_work = L
    while ids.shape[0] > 0:
        # loop condition
        cond = (prev_end - prev_beg >= 1) | (leftover != -1)
        if not cond.all():
            keep = cond
            ids, prev_beg, prev_end, leftover = ids[keep], prev_beg[keep], prev_end[keep], leftover[keep]
            if ids.shape[0] == 0:
                break
        width = prev_end - prev_beg
        length = width // 2
        length = length + ((width % 2 == 1) & (leftover != -1) & (leftover != prev_end - 1))
        i, owner = _ragged_arange(length)
        c0 = prev_beg[owner] + 2 * i
        c1 = c0 + 1
        use_left = c1 >= prev_end[owner]
        if use_left.any():
            assert (leftover[owner[use_left]] != -1).all()
            c1[use_left] = leftover[owner[use_left]]
            consumed = np.zeros(ids.shape[0], dtype = bool)
            consumed[owner[use_left]] = True
            leftover = np.where(consumed, -1, leftover)
        parent = prev_end[owner] + i
        b = ids[owner]
        cnt = parent.shape[0]
        work_parent = next_work + np.arange(cnt, dtype = np.int64)
        next_work += cnt
        w0 = final_to_work[base[b] + c0]
        w1 = final_to_work[base[b] + c1]
        assert (w0 >= 0).all() and (w1 >= 0).all()
        final_to_work[base[b] + parent] = work_parent
        levels.append((w0.astype(np.int32), w1.astype(np.int32)))
        children.append(np.stack([c0, c1], axis = 1).astype(np.int32))
        # break / update
        brk = (length == 1) & (leftover == -1)
        prev_beg = prev_end
        prev_end = prev_beg + length
        leftover = np.where((length % 2 == 1) & (leftover == -1), prev_end - 1, leftover)
        keep = ~brk
        ids, prev_beg, prev_end, leftover = ids[keep], prev_beg[keep], prev_end[keep], leftover[keep]

    assert next_work == N and (final_to_work >= 0).all()
    plan = _Plan()
    plan.num_leaves = nl.astype(np.int32)
    plan.base = base.astype(np.int32)
    plan.leaf_start = leaf_start.astype(np.int32)
    plan.num_nodes = N
    plan.num_leaves_total = L
    plan.levels = levels
    plan.final_to_work = final_to_work.astype(np.int32)
    plan.internal_children = (np.concatenate(children, axis = 0) if children
                              else np.zeros((0, 2), dtype = np.int32))
    plan.mx_levels = [(mx.array(a), mx.array(b)) for a, b in levels]
    plan.mx_final_to_work = mx.array(plan.final_to_work)
    plan.mx_internal_children = mx.array(plan.internal_children)
    plan.mx_leaf_bvh = mx.array(np.repeat(np.arange(B, dtype = np.int64), nl))
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = plan
    return plan


def _key32(sort_key):
    """ Order-preserving map of a sort key to int64 values in [0, 2^32). """
    k = sort_key if isinstance(sort_key, mx.array) else mx.array(np.asarray(sort_key))
    k = k.reshape(-1)
    if k.dtype in (mx.float16, mx.float32, mx.float64, mx.bfloat16):
        k = k.astype(mx.float32)
        # Canonicalise -0.0 to +0.0 so that equal keys compare equal (as `<` does).
        k = mx.where(k == 0, mx.array(0.0, dtype = mx.float32), k)
        bits = mx.view(k, mx.uint32).astype(mx.int64)
        mag = bits & 0x7FFFFFFF
        neg = (bits >> 31) != 0
        # negative: larger magnitude sorts first; positive: after all negatives.
        return mx.where(neg, 0x7FFFFFFF - mag, mag + 0x80000000)
    if k.dtype in (mx.int8, mx.int16, mx.int32):
        return k.astype(mx.int64) + 0x80000000
    # unsigned or int64 keys: must lie in [0, 2^32)
    return k.astype(mx.int64)


def _merge(a, b):
    """
        Parent (N, 5) rows from child rows a (child0), b (child1), with the C++
        tie semantics: diffvg min(x, y) = x < y ? x : y and max(x, y) =
        x > y ? x : y (so merge returns child1 on ties); std::max(r0, r1) =
        r0 < r1 ? r1 : r0 (child0 on ties).
    """
    lo = mx.where(a[:, :2] < b[:, :2], a[:, :2], b[:, :2])
    hi = mx.where(a[:, 2:4] > b[:, 2:4], a[:, 2:4], b[:, 2:4])
    r = mx.where(a[:, 4:] < b[:, 4:], b[:, 4:], a[:, 4:])
    return mx.concatenate([lo, hi, r], axis = 1)


def _evaluate_levels(plan, leaves_sorted):
    """ leaves_sorted (L, 5) in work order -> nodes_f (N, 5) in pool layout """
    L = plan.num_leaves_total
    N = plan.num_nodes
    if N == L:
        return leaves_sorted[plan.mx_final_to_work]
    work = mx.concatenate([leaves_sorted, mx.zeros((N - L, 5), dtype = mx.float32)], axis = 0)
    start = L
    for w0, w1 in plan.mx_levels:
        vals = _merge(work[w0], work[w1])
        work[start:start + vals.shape[0]] = vals
        start += vals.shape[0]
    return work[plan.mx_final_to_work]


def _as_mx(a, dtype):
    if isinstance(a, mx.array):
        return a if a.dtype == dtype else a.astype(dtype)
    return mx.array(np.asarray(a), dtype = dtype)


def build_bvhs(leaf_boxes, leaf_radii, leaf_payload, leaf_bvh, sort_key, num_leaves_per_bvh):
    """
        Builds B BVHs at once.
        leaf_boxes   (L, 4) float32 [p_min.x, p_min.y, p_max.x, p_max.y]
        leaf_radii   (L,) float32
        leaf_payload (L, 2) int32: (child0, child1) leaf encoding
        leaf_bvh     (L,) int: BVH of each leaf, nondecreasing (leaves of a BVH
                     contiguous). Only its consistency with num_leaves_per_bvh
                     matters: the plan derives the BVH ids from the counts.
        sort_key     (L,) float (compared as float32) or integer in [0, 2^32)
                     (int8/16/32 also accepted, compared signed); None keeps the
                     input order (build_bvh<false> on pre-sorted leaves).
        num_leaves_per_bvh (B,) host ints, all >= 1.
        Returns BVHBuild (lazy MLX arrays; nothing is evaluated here).
    """
    plan = build_plan(num_leaves_per_bvh)
    L = plan.num_leaves_total
    boxes = _as_mx(leaf_boxes, mx.float32).reshape(L, 4)
    radii = _as_mx(leaf_radii, mx.float32).reshape(L, 1)
    payload = _as_mx(leaf_payload, mx.int32).reshape(L, 2)
    del leaf_bvh  # implied by num_leaves_per_bvh (see docstring)
    if sort_key is None:
        perm = mx.arange(L, dtype = mx.int32)
    else:
        composite = (plan.mx_leaf_bvh << 32) | _key32(sort_key)
        perm = mx.argsort(composite).astype(mx.int32)
    leaves = mx.concatenate([boxes, radii], axis = 1)[perm]
    nodes_f = _evaluate_levels(plan, leaves)
    ints_work = mx.concatenate([payload[perm], plan.mx_internal_children], axis = 0)
    nodes_i = ints_work[plan.mx_final_to_work]
    root = (plan.base + 2 * plan.num_leaves - 2).astype(np.int32)
    return BVHBuild(nodes_f, nodes_i, plan.base, root, perm, plan)


def refit_bvhs(build, leaf_boxes, leaf_radii):
    """
        Recomputes all boxes and radii of `build` (same topology and leaf
        order) from new leaf values given in the ORIGINAL leaf order.
        Returns nodes_f mx.float32 (N_total, 5). nodes_i is unchanged.
    """
    plan = build.plan
    L = plan.num_leaves_total
    boxes = _as_mx(leaf_boxes, mx.float32).reshape(L, 4)
    radii = _as_mx(leaf_radii, mx.float32).reshape(L, 1)
    leaves = mx.concatenate([boxes, radii], axis = 1)[build.perm]
    return _evaluate_levels(plan, leaves)


def _expand_bits(x):
    """ diffvg.h expand_bits: bits 0..9 of x interleaved with zeros (x uint32) """
    r = mx.zeros_like(x)
    for i in range(10):
        r = r | ((x & (1 << i)) << i)
    return r


def morton2d_keys(centres, canvas_width, canvas_height):
    """
        scene.cpp morton2D for (L, 2) centres -> (L,) uint32.
        pp = centre / (canvas_width, canvas_height) in float32, then
        TVector2<uint32_t>{pp.x * 1023, pp.y * 1023} and
        (expand_bits(x) << 1) | expand_bits(y).

        float -> uint32 conversion: C++ leaves out-of-range values undefined;
        on arm64 (the platform of this backend) clang emits fcvtzu, which
        truncates toward zero and saturates: NaN -> 0, negative -> 0,
        >= 2^32 (incl. +inf) -> 0xFFFFFFFF. That is replicated here explicitly
        (not relying on the Metal cast). expand_bits then keeps only the low
        10 bits, so e.g. a centre slightly right of the canvas (pp.x*1023 =
        1030) maps to 1030 & 1023 = 6, exactly as the CPU does.
    """
    c = _as_mx(centres, mx.float32).reshape(-1, 2)
    wh = mx.array([float(canvas_width), float(canvas_height)], dtype = mx.float32)
    v = (c / wh) * mx.array(1023.0, dtype = mx.float32)
    v = mx.where(mx.isnan(v), mx.array(0.0, dtype = mx.float32), v)
    v = mx.maximum(v, mx.array(0.0, dtype = mx.float32))
    big = v >= mx.array(4294967296.0, dtype = mx.float32)
    vi = mx.where(big, mx.array(0xFFFFFFFF, dtype = mx.uint32),
                  mx.floor(mx.where(big, mx.array(0.0, dtype = mx.float32), v)).astype(mx.uint32))
    vi = vi & mx.array(1023, dtype = mx.uint32)
    x = _expand_bits(vi[:, 0])
    y = _expand_bits(vi[:, 1])
    return (x << mx.array(1, dtype = mx.uint32)) | y
