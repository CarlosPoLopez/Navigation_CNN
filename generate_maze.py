"""Generates random mazes to feed the network, saved as a 2D .npy (1 = wall, 0 = corridor).

The output is directly compatible with predict.py (which reads a raw 2D .npy with
`--wall auto`, treating >0.5 as wall). Entry and exit sit at opposite corners
(top-left and bottom-right), matching the goal-conditioning markers of predict.py, so a
generated maze can be solved right away:

    python generate_maze.py --out maze.npy
    python predict.py --input maze.npy --ckpt_dir runs/final_model --k 1

Two maze types:
  - routes  (default): a maze built to expose N vertex-disjoint paths between entry and
    exit (the distribution the network was trained on). No wall islands by construction.
  - perfect          : a perfect maze (DFS backtracking): exactly one path between any
    two cells, no loops.
"""

import os
import argparse
from collections import deque

import numpy as np


def exact_dimx(nc, cell, wall_thickness=3):
    """Exact side (px) so the grid of nc cells tiles the image with no leftover border."""
    return nc * (cell + wall_thickness) + wall_thickness


# -- Perfect maze (DFS backtracking) ---------------------------------------------

def perfect_maze(dimx=501, cell=50, wall_thickness=3, seed=None):
    """Perfect maze via iterative DFS backtracking: exactly one path between any two
    cells, no loops, fully solvable, and no floating wall islands (the wall set is a
    spanning tree). Returns a (dimx, dimx) int array: 1 = wall, 0 = corridor."""
    if seed is not None:
        np.random.seed(seed)

    gp = wall_thickness
    nc = (dimx - gp) // (cell + gp)
    cell = (dimx - gp * (nc + 1)) // nc              # fit the cells to the available space
    dimx_real = nc * (cell + gp) + gp
    offset = (dimx - dimx_real) // 2

    def px(k):                                       # top-left pixel of cell index k
        return offset + gp + k * (cell + gp)

    aa = np.ones((dimx, dimx), dtype=int)            # start all walls
    visited = np.zeros((nc, nc), dtype=bool)

    r0, c0 = 0, 0
    aa[px(r0):px(r0) + cell, px(c0):px(c0) + cell] = 0
    visited[r0, c0] = True
    stack = [(r0, c0)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in directions
                if 0 <= r + dr < nc and 0 <= c + dc < nc and not visited[r + dr, c + dc]]
        if nbrs:
            nr, ncc = nbrs[np.random.randint(len(nbrs))]
            # carve the wall between (r,c) and (nr,ncc)
            if nr == r:                              # same row -> vertical wall between columns
                jmin = min(c, ncc)
                aa[px(r):px(r) + cell, px(jmin) + cell:px(jmin) + cell + gp] = 0
            else:                                    # same column -> horizontal wall between rows
                imin = min(r, nr)
                aa[px(imin) + cell:px(imin) + cell + gp, px(c):px(c) + cell] = 0
            aa[px(nr):px(nr) + cell, px(ncc):px(ncc) + cell] = 0
            visited[nr, ncc] = True
            stack.append((nr, ncc))
        else:
            stack.pop()                              # backtrack

    aa[:gp, :] = 1; aa[-gp:, :] = 1; aa[:, :gp] = 1; aa[:, -gp:] = 1   # seal the border
    return aa


# -- Multi-route maze (N vertex-disjoint paths entry->exit) -----------------------

def multi_route_maze(dimx=590, cell=60, wall_thickness=5,
                     n_routes=2, start_cell=None, end_cell=None, seed=None):
    """Builds a maze guaranteeing N vertex-disjoint paths from start to exit (they share
    no interior cell, so they are physically distinct routes of different length — a
    non-trivial shortest-path decision). No wall islands by construction. This is the
    distribution the network was trained on. Returns (dimx, dimx) int: 1 = wall, 0 = corridor.

    Algorithm: (1) find N vertex-disjoint cell paths by sequential BFS/DFS; (2) open their
    corridors; (3) grow dead-end DFS trees from the path cells to cover the rest of the
    grid (without connecting distinct routes, so no shortcuts); (4) seal the border.
    On a grid with start/exit at opposite corners the theoretical max is 2 disjoint paths.
    """
    if seed is not None:
        np.random.seed(seed)

    gp = wall_thickness
    nc = (dimx - gp) // (cell + gp)
    cell = (dimx - gp * (nc + 1)) // nc
    dimx_real = nc * (cell + gp) + gp
    offset = (dimx - dimx_real) // 2

    def px(k):
        return offset + gp + k * (cell + gp)

    if start_cell is None:
        start_cell = (0, 0)
    if end_cell is None:
        end_cell = (nc - 1, nc - 1)

    def neighbors(rc):
        r, c = rc
        return [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < nc and 0 <= c + dc < nc]

    def bfs_path(blocked):                           # shortest route (short path)
        parent = {start_cell: None}
        queue = deque([start_cell])
        while queue:
            u = queue.popleft()
            if u == end_cell:
                break
            vs = [v for v in neighbors(u)
                  if v not in parent and (v == end_cell or v not in blocked)]
            np.random.shuffle(vs)
            for v in vs:
                parent[v] = u
                queue.append(v)
        if end_cell not in parent:
            return None
        path, cur = [], end_cell
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return path[::-1]

    def dfs_path(blocked):                           # randomized DFS -> winding, long route
        parent = {start_cell: None}
        visited = {start_cell}
        stack = [start_cell]
        while stack:
            u = stack[-1]
            if u == end_cell:
                break
            vs = [v for v in neighbors(u)
                  if v not in visited and (v == end_cell or v not in blocked)]
            np.random.shuffle(vs)
            if vs:
                v = vs[0]
                parent[v] = u; visited.add(v); stack.append(v)
            else:
                stack.pop()
        if end_cell not in parent:
            return None
        path, cur = [], end_cell
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return path[::-1]

    # Retry the short path until a distinct, sufficiently longer second path exists
    # (a bad first path can wall start off from end and block the second one).
    min_extra = 6
    routes = []
    for _outer in range(30):
        p1 = bfs_path(set())
        if p1 is None:
            continue
        blocked = set(p1[1:-1])
        best_p2 = None
        for _ in range(60):
            c = dfs_path(blocked)
            if c is None:
                continue
            if best_p2 is None or len(c) > len(best_p2):
                best_p2 = c
            if best_p2 is not None and len(best_p2) >= len(p1) + min_extra:
                break
        if best_p2 is not None and len(best_p2) >= len(p1) + min_extra:
            routes = [p1, best_p2]
            break
    if not routes:                                   # fall back to whatever we can get
        p1 = bfs_path(set())
        routes = [p1] if p1 is not None else []
        if p1 is not None:
            c = dfs_path(set(p1[1:-1]))
            if c is not None:
                routes.append(c)

    route_cells = set()
    for route in routes:
        route_cells.update(route)

    aa = np.ones((dimx, dimx), dtype=int)

    def open_cell(rc):
        r, c = rc
        aa[px(r):px(r) + cell, px(c):px(c) + cell] = 0

    def open_wall(a, b):
        ra, ca = a
        rb, cb = b
        if ra == rb:                                 # horizontal neighbors
            jmin = min(ca, cb)
            aa[px(ra):px(ra) + cell, px(jmin) + cell:px(jmin) + cell + gp] = 0
        else:                                        # vertical neighbors
            imin = min(ra, rb)
            aa[px(imin) + cell:px(imin) + cell + gp, px(ca):px(ca) + cell] = 0

    for route in routes:
        for c in route:
            open_cell(c)
        for i in range(len(route) - 1):
            open_wall(route[i], route[i + 1])

    # Fill the rest with dead-end trees hanging off the route cells (no cross-links).
    visited = set(route_cells)
    stack = list(route_cells)
    np.random.shuffle(stack)
    while stack:
        u = stack[-1]
        candidates = [v for v in neighbors(u) if v not in visited]
        if candidates:
            v = candidates[np.random.randint(len(candidates))]
            open_cell(v); open_wall(u, v); visited.add(v); stack.append(v)
        else:
            stack.pop()

    aa[:gp, :] = 1; aa[-gp:, :] = 1; aa[:, :gp] = 1; aa[:, -gp:] = 1   # seal the border
    return aa


# -- CLI -------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate a random maze as a 2D .npy (1=wall, 0=corridor).')
    p.add_argument('--type', choices=['routes', 'perfect'], default='routes',
                   help="'routes' (default, the trained distribution) or 'perfect'.")
    p.add_argument('--nc', type=int, default=9, help='cells per side (9=normal, 5=simple).')
    p.add_argument('--cell', type=int, default=60, help='cell size in pixels.')
    p.add_argument('--wall', type=int, default=5, help='wall thickness in pixels.')
    p.add_argument('--seed', type=int, default=None, help='random seed (reproducible).')
    p.add_argument('--out', default='maze.npy', help='output .npy path.')
    p.add_argument('--png', action='store_true', help='also save a preview PNG next to it.')
    args = p.parse_args()

    dimx = exact_dimx(nc=args.nc, cell=args.cell, wall_thickness=args.wall)
    if args.type == 'perfect':
        maze = perfect_maze(dimx=dimx, cell=args.cell, wall_thickness=args.wall, seed=args.seed)
    else:
        maze = multi_route_maze(dimx=dimx, cell=args.cell, wall_thickness=args.wall,
                                n_routes=2, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, maze)
    print(f'{args.type} maze  nc={args.nc}  {dimx}x{dimx}  -> {args.out}')

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(maze, cmap='binary', origin='lower'); ax.axis('off')
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + '.png'
        fig.savefig(png, dpi=130); plt.close(fig)
        print(f'preview -> {png}')

