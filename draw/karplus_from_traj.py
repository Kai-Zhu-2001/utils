#!/usr/bin/env python3
import argparse
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.data import covalent_radii, atomic_numbers
import collections
import os

# -------------------- math helpers --------------------
def dihedral_angle(p0, p1, p2, p3):
    """Return dihedral angle (radians) defined by four 3D points (ASE style)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.arctan2(y, x)  # (-pi, pi)

def block_std_err(values, weights=None, nblock=20):
    """Block-averaging standard error of the mean."""
    n = len(values)
    if nblock > n:
        nblock = max(1, n // 2)
    m = n // nblock
    if m == 0:
        return np.nan
    vals = np.asarray(values, float)
    blocks = []
    if weights is None:
        for i in range(nblock):
            block = vals[i*m:(i+1)*m]
            if len(block) == 0:
                continue
            blocks.append(block.mean())
    else:
        w = np.asarray(weights, float)
        for i in range(nblock):
            sl = slice(i*m, (i+1)*m)
            vw, ww = vals[sl], w[sl]
            s = ww.sum()
            if s <= 0 or len(vw) == 0:
                continue
            blocks.append(np.average(vw, weights=ww))
    if len(blocks) <= 1:
        return np.nan
    return np.std(blocks, ddof=1) / np.sqrt(len(blocks))

def karplus(phi_radians, A, B, C, form="Acos2+Bcos+C"):
    """Karplus mapping. phi in radians."""
    if form == "Acos2+Bcos+C":
        return A * np.cos(phi_radians)**2 + B * np.cos(phi_radians) + C
    elif form == "Acos2(2phi)+Bcos(phi)+C":
        return A * np.cos(2*phi_radians) + B * np.cos(phi_radians) + C
    else:
        raise ValueError("Unknown Karplus form")

# -------------------- graph / topology --------------------
def validate_quartet(indices, natoms):
    for idx in indices:
        if idx < 0 or idx >= natoms:
            raise ValueError(f"Atom index {idx} out of range [0, {natoms-1}]")

def build_bond_graph(atoms, bond_mult=1.1, bond_skin=0.3):
    """Undirected adjacency list using ASE NeighborList with natural cutoffs."""
    cutoffs = natural_cutoffs(atoms, mult=bond_mult)
    nl = NeighborList(cutoffs, skin=bond_skin, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for i in range(n):
        idxs, _ = nl.get_neighbors(i)
        for j in idxs:
            if i == j:
                continue
            adj[i].add(j)
            adj[j].add(i)
    return adj, nl

def connected_components(adj):
    n = len(adj)
    seen = [False]*n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

def enumerate_dihedrals_from_graph(adj):
    """
    Enumerate unique dihedrals i-j-k-m where j-k is a bond,
    i is bonded to j (i != k), m is bonded to k (m != j), and all four distinct.
    (i,j,k,m) ~ (m,k,j,i), keep lexicographically smaller.
    """
    n = len(adj)
    seen = set()
    dihs = []
    for j in range(n):
        for k in adj[j]:
            if j >= k:
                continue
            for i in adj[j]:
                if i == k:
                    continue
                for m in adj[k]:
                    if m == j:
                        continue
                    if len({i, j, k, m}) < 4:
                        continue
                    tup = (i, j, k, m)
                    rev = (m, k, j, i)
                    key = min(tup, rev)
                    if key in seen:
                        continue
                    seen.add(key)
                    dihs.append(tup)
    return dihs

def quartet_elements(atoms, quartet):
    idx1, idx2, idx3, idx4 = quartet
    syms = atoms.get_chemical_symbols()
    return f"{syms[idx1]}-{syms[idx2]}-{syms[idx3]}-{syms[idx4]}"

# --------- optional: valence sanity prune (lightweight) ----------
DEFAULT_MAX_DEG = {'H': 1, 'O': 2, 'N': 3, 'C': 4}
def sanity_prune_valence(adj, symbols, max_deg=DEFAULT_MAX_DEG, slack=1):
    """
    Drop edges greedily if a node exceeds (max_deg+slack).
    Note: without edge-length priorities this is conservative but effective to remove outliers.
    """
    n = len(adj)
    adj = [set(nei) for nei in adj]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            lim = max_deg.get(symbols[i], 6) + slack
            while len(adj[i]) > lim:
                # remove one neighbor (arbitrary). Optionally plug in distance-based selection if needed.
                j = next(iter(adj[i]))
                adj[i].remove(j); adj[j].remove(i)
                changed = True
    return adj

# -------------------- Ala peptide heuristics --------------------
def _has_oxygen_neighbor(adj, atoms, idx):
    syms = atoms.get_chemical_symbols()
    return any(syms[j] == 'O' for j in adj[idx])

def find_backbone_triplets_N_CA_Cprime(adj, atoms, keep_set=None):
    """Return list of (N, CA, C') triplets using minimal chemistry heuristics."""
    syms = atoms.get_chemical_symbols()
    triplets = []
    for ca in range(len(atoms)):
        if syms[ca] != 'C':
            continue
        if keep_set is not None and ca not in keep_set:
            continue
        Ns = [n for n in adj[ca] if syms[n] == 'N' and (keep_set is None or n in keep_set)]
        if not Ns:
            continue
        carbons = [c for c in adj[ca] if syms[c] == 'C' and (keep_set is None or c in keep_set)]
        Cprime = None
        for cc in carbons:
            if _has_oxygen_neighbor(adj, atoms, cc):
                Cprime = cc
                break
        if Cprime is None:
            continue
        for Nidx in Ns:
            triplets.append((Nidx, ca, Cprime))
    uniq, seen = [], set()
    for t in triplets:
        if t in seen:
            continue
        seen.add(t); uniq.append(t)
    return uniq

def build_ala_3J_quartets(adj, atoms, keep_set=None):
    """
    Return list of ((i,j,k,m), type_tag) for HNHα, HNCβ, HNC' across all residues.
    j-k is N-CA.
    """
    syms = atoms.get_chemical_symbols()
    out = []
    triplets = find_backbone_triplets_N_CA_Cprime(adj, atoms, keep_set)
    for Nidx, CA, Cprime in triplets:
        HNs = [h for h in adj[Nidx] if syms[h] == 'H' and (keep_set is None or h in keep_set)]
        HAs = [h for h in adj[CA] if syms[h] == 'H' and (keep_set is None or h in keep_set)]
        CBs = [c for c in adj[CA] if syms[c] == 'C' and c not in (Nidx, Cprime) and (keep_set is None or c in keep_set)]
        for hn in HNs:
            for ha in HAs:
                if len({hn, Nidx, CA, ha}) == 4:
                    out.append(((hn, Nidx, CA, ha), "HNHα"))
            for cb in CBs:
                if len({hn, Nidx, CA, cb}) == 4:
                    out.append(((hn, Nidx, CA, cb), "HNCβ"))
            if len({hn, Nidx, CA, Cprime}) == 4:
                out.append(((hn, Nidx, CA, Cprime), "HNC'"))
    uniq, seen = [], set()
    for (i,j,k,m), tag in out:
        key = min((i,j,k,m), (m,k,j,i))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(((i,j,k,m), tag))
    return uniq

# -------------------- optional utils --------------------
def load_solute_idx(path):
    arr = np.loadtxt(path, dtype=int)
    if arr.ndim == 0:
        arr = np.array([int(arr)])
    return set(arr.tolist())

# -------------------- CLI & main --------------------
def main():
    p = argparse.ArgumentParser(description="Compute 3J via Karplus from ASE .traj (Ala peptide friendly)")
    p.add_argument("--traj", required=True, help="ASE trajectory file, e.g., prot.traj")
    p.add_argument("--quad", nargs="+", help="Manual quartets: i,j,k,m (0-based)")
    p.add_argument("--auto", action="store_true", help="Auto-enumerate ALL dihedrals (after filtering)")
    p.add_argument("--auto_ala", action="store_true",
                   help="Auto-detect Ala peptide ³J quartets: HN–Hα, HN–Cβ, HN–C′")
    p.add_argument("--bond_mult", type=float, default=1.1, help="Neighbor cutoff multiplier (natural_cutoffs*mult)")
    p.add_argument("--bond_skin", type=float, default=0.3, help="NeighborList skin (Å)")
    p.add_argument("--valence_prune", action="store_true", help="Apply lightweight valence-based pruning on the bond graph")
    p.add_argument("--solute_idx_file", default=None,
                   help="Text file of solute atom indices (0-based), one or multiple per line; overrides largest-component filter")
    # global Karplus (fallback for all)
    p.add_argument("--A", type=float, default=6.51, help="Karplus A (Hz) [fallback]")
    p.add_argument("--B", type=float, default=-1.76, help="Karplus B (Hz) [fallback]")
    p.add_argument("--C", type=float, default=1.60, help="Karplus C (Hz) [fallback]")
    p.add_argument("--form", default="Acos2+Bcos+C",
                   choices=["Acos2+Bcos+C", "Acos2(2phi)+Bcos(phi)+C"], help="Karplus functional form")
    # global theta (degrees)
    p.add_argument("--theta", type=float, default=0.0,
                   help="Global theta in degrees; phi_eff = phi + theta (fallback for all types)")
    # per-type Karplus (optional)
    p.add_argument("--per_type_params", action="store_true",
                   help="Use per-type Karplus parameters for HNHα/HNCβ/HNC′ if provided below")
    p.add_argument("--A_HNHalpha", type=float, default=None)
    p.add_argument("--B_HNHalpha", type=float, default=None)
    p.add_argument("--C_HNHalpha", type=float, default=None)
    p.add_argument("--A_HNCb", type=float, default=None)
    p.add_argument("--B_HNCb", type=float, default=None)
    p.add_argument("--C_HNCb", type=float, default=None)
    p.add_argument("--A_HNCp", type=float, default=None)  # C prime
    p.add_argument("--B_HNCp", type=float, default=None)
    p.add_argument("--C_HNCp", type=float, default=None)
    # per-type thetas (degrees)
    p.add_argument("--per_type_thetas", action="store_true",
                   help="Use per-type theta (deg) for HNHα/HNCβ/HNC′ if provided below")
    p.add_argument("--theta_HNHalpha", type=float, default=None, help="theta for HN–Hα (deg)")
    p.add_argument("--theta_HNCb",     type=float, default=None, help="theta for HN–Cβ (deg)")
    p.add_argument("--theta_HNCp",     type=float, default=None, help="theta for HN–C′ (deg)")
    # sampling / IO
    p.add_argument("--weights", default=None, help="weights.txt (per original frame)")
    p.add_argument("--skip", type=int, default=1, help="Stride frames")
    p.add_argument("--nblock", type=int, default=20, help="Blocks for SE")
    p.add_argument("--out", default=None, help="npz basename for per-frame arrays (without extension is fine)")
    p.add_argument("--degrees", action="store_true", help="Print mean phi in degrees")
    p.add_argument("--no_keep_largest", action="store_true",
                   help="Do NOT restrict to largest connected component (ignored if --solute_idx_file is set)")
    p.add_argument("--max_frames", type=int, default=40000, help="Max frames to read from traj (default 20000)")
    args = p.parse_args()

    # load trajectory
    traj = read(args.traj, index=f":{args.max_frames}")
    if len(traj) == 0:
        raise ValueError("No frames found in trajectory file")
    frames = traj[::args.skip]
    nF = len(frames); natoms = len(frames[0])
    print(f"[INFO] Loaded {len(traj)} frames, used {nF} frames (skip={args.skip})")
    print(f"[INFO] System has {natoms} atoms")

    # neighbor graph on first frame
    adj, _ = build_bond_graph(frames[0], bond_mult=args.bond_mult, bond_skin=args.bond_skin)

    # optional valence-prune (lightweight sanity)
    if args.valence_prune:
        syms0 = frames[0].get_chemical_symbols()
        adj = sanity_prune_valence(adj, syms0)
        print("[INFO] Applied valence-based pruning to the bond graph")

    # decide keep_set
    keep_set = None
    if args.solute_idx_file is not None and os.path.exists(args.solute_idx_file):
        keep_set = load_solute_idx(args.solute_idx_file)
        print(f"[INFO] Using solute index file with {len(keep_set)} atoms (overrides largest-component filter)")
    elif not args.no_keep_largest:
        comps = connected_components(adj)
        sizes = [len(c) for c in comps]
        main_comp = comps[int(np.argmax(sizes))]
        keep_set = set(main_comp)
        print(f"[INFO] Keeping largest component: {len(main_comp)} atoms; excluding {natoms-len(main_comp)} others")
    else:
        print("[WARN] No keep-set applied (you disabled largest-component filter and did not provide solute index file)")

    # collect quartets
    typed_quartets = []  # list of (quartet, type_tag)
    quartets = []

    # manual quartets
    if args.quad:
        for q in args.quad:
            idx = [int(x) for x in q.split(",")]
            if len(idx) != 4:
                raise ValueError(f"Quartet must be i,j,k,m: got {q}")
            validate_quartet(idx, natoms)
            quartets.append(tuple(idx))

    # Ala-specific automatic quartets
    if args.auto_ala:
        ala_quads = build_ala_3J_quartets(adj, frames[0], keep_set)
        print(f"[INFO] Detected {len(ala_quads)} Ala ³J quartets (HN–Hα / HN–Cβ / HN–C′).")
        for q, tag in ala_quads:
            quartets.append(q)
            typed_quartets.append((q, tag))

    # generic dihedrals
    if args.auto:
        auto_quads = enumerate_dihedrals_from_graph(adj)
        if keep_set is not None:
            auto_quads = [q for q in auto_quads if all(idx in keep_set for idx in q)]
        print(f"[INFO] Found {len(auto_quads)} generic dihedrals after filtering.")
        quartets += auto_quads

    if not quartets:
        raise ValueError("No quartets to process. Use --auto_ala / --auto / --quad.")

    # deduplicate and apply keep_set filter
    seen = set(); qlist = []
    for (i,j,k,m) in quartets:
        key = min((i,j,k,m), (m,k,j,i))
        if key in seen:
            continue
        if keep_set is not None and not all(idx in keep_set for idx in (i,j,k,m)):
            continue
        seen.add(key); qlist.append((i,j,k,m))
    quartets = qlist
    print(f"[INFO] Processing {len(quartets)} unique quartets")

    # weights
    w = None
    if args.weights is not None:
        allw = np.loadtxt(args.weights, dtype=float)
        if len(allw) != len(traj):
            raise ValueError(f"weights length ({len(allw)}) must equal original number of frames ({len(traj)})")
        w = allw[::args.skip]
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(w > 0):
            raise ValueError("at least one weight must be positive")
        print(f"[INFO] Using weights: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")

    # build type lookup
    tag_lookup = {}
    for q, tag in typed_quartets:
        key = min(q, (q[3], q[2], q[1], q[0]))
        tag_lookup[key] = tag

    # per-type params (optional)
    def get_params_for_tag(tag):
        if (not args.per_type_params) or tag not in ("HNHα", "HNCβ", "HNC'"):
            return args.A, args.B, args.C  # fallback
        if tag == "HNHα" and args.A_HNHalpha is not None and args.B_HNHalpha is not None and args.C_HNHalpha is not None:
            return args.A_HNHalpha, args.B_HNHalpha, args.C_HNHalpha
        if tag == "HNCβ" and args.A_HNCb is not None and args.B_HNCb is not None and args.C_HNCb is not None:
            return args.A_HNCb, args.B_HNCb, args.C_HNCb
        if tag == "HNC'" and args.A_HNCp is not None and args.B_HNCp is not None and args.C_HNCp is not None:
            return args.A_HNCp, args.B_HNCp, args.C_HNCp
        return args.A, args.B, args.C  # fallback if incomplete

    # per-type theta (deg) with fallback to global
    def get_theta_for_tag(tag):
        if (not args.per_type_thetas) or tag not in ("HNHα", "HNCβ", "HNC'"):
            return np.deg2rad(args.theta)
        if tag == "HNHα" and args.theta_HNHalpha is not None:
            return np.deg2rad(args.theta_HNHalpha)
        if tag == "HNCβ" and args.theta_HNCb is not None:
            return np.deg2rad(args.theta_HNCb)
        if tag == "HNC'" and args.theta_HNCp is not None:
            return np.deg2rad(args.theta_HNCp)
        return np.deg2rad(args.theta)

    # compute per-quartet time series
    all_phis, all_J = [], []
    results, quartet_elem, type_tags = [], [], []
    first_atoms = frames[0]

    for qid, (i, j, k, m) in enumerate(quartets):
        phis = np.empty(nF, dtype=float)
        for t, at in enumerate(frames):
            pos = at.get_positions()
            phis[t] = dihedral_angle(pos[i], pos[j], pos[k], pos[m])

        key = min((i,j,k,m), (m,k,j,i))
        this_type = tag_lookup.get(key, "generic")
        A_use, B_use, C_use = get_params_for_tag(this_type)
        theta_use = get_theta_for_tag(this_type)
        phis_eff = phis + theta_use

        J = karplus(phis_eff, A_use, B_use, C_use, form=args.form)

        if w is None:
            J_mean = J.mean()
            phi_mean = np.arctan2(np.mean(np.sin(phis)), np.mean(np.cos(phis)))
        else:
            J_mean = np.average(J, weights=w)
            phi_mean = np.arctan2(np.average(np.sin(phis), weights=w),
                                  np.average(np.cos(phis), weights=w))
        J_se = block_std_err(J, weights=w, nblock=args.nblock)

        elem_tag = quartet_elements(first_atoms, (i,j,k,m))
        type_tags.append(this_type)
        quartet_elem.append(elem_tag)
        all_phis.append(phis)
        all_J.append(J)

        if args.degrees:
            phi_mean_deg = np.rad2deg(phi_mean)
            phi_std_deg = np.rad2deg(np.std(phis))
            print(f"[RESULT] #{qid:04d} {i}-{j}-{k}-{m} [{elem_tag}] <{this_type}>: "
                  f"<J>={J_mean:.2f} ± {J_se:.2f} Hz, <phi>={phi_mean_deg:.1f}° ± {phi_std_deg:.1f}° (n={nF})")
        else:
            print(f"[RESULT] #{qid:04d} {i}-{j}-{k}-{m} [{elem_tag}] <{this_type}>: "
                  f"<J>={J_mean:.2f} ± {J_se:.2f} Hz, <phi>={phi_mean:.3f} ± {np.std(phis):.3f} rad (n={nF})")

        results.append((qid, (i, j, k, m), elem_tag, this_type, J_mean, J_se, phi_mean))

    # save arrays
    if args.out:
        outpath = args.out if args.out.endswith(".npz") else args.out + ".npz"
        save_dict = {
            'quartets': np.array(quartets, dtype=int),
            'quartet_elem': np.array(quartet_elem, dtype=object),
            'type_tags': np.array(type_tags, dtype=object),
            'phis': np.array(all_phis, dtype=object),
            'J': np.array(all_J, dtype=object),
            'karplus_form': args.form,
            'karplus_fallback': np.array([args.A, args.B, args.C]),
            'theta_fallback_deg': args.theta,
            'per_type_params': {
                'HNHalpha': (args.A_HNHalpha, args.B_HNHalpha, args.C_HNHalpha),
                'HNCb':     (args.A_HNCb,     args.B_HNCb,     args.C_HNCb),
                'HNCp':     (args.A_HNCp,     args.B_HNCp,     args.C_HNCp),
            } if args.per_type_params else None,
            'per_type_thetas_deg': {
                'HNHalpha': args.theta_HNHalpha,
                'HNCb':     args.theta_HNCb,
                'HNCp':     args.theta_HNCp,
            } if args.per_type_thetas else None,
        }
        if w is not None:
            save_dict['weights'] = w
        np.savez(outpath, **save_dict)
        print(f"[INFO] Saved arrays to {outpath}")
        summary_file = outpath.replace('.npz', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("# NMR 3J analysis summary (Ala peptide)\n")
            f.write(f"# Trajectory: {args.traj}\n")
            f.write(f"# Frames used: {nF} (skip={args.skip})\n")
            f.write(f"# Karplus form: {args.form}\n")
            f.write(f"# Fallback A,B,C: {args.A},{args.B},{args.C}\n")
            f.write(f"# Fallback theta (deg): {args.theta}\n")
            f.write(f"# Per-type params used: {args.per_type_params}\n")
            f.write(f"# Per-type thetas used: {args.per_type_thetas}\n")
            f.write(f"# Keep largest component: {not args.no_keep_largest}\n")
            f.write(f"# Solute index file: {args.solute_idx_file}\n")
            f.write("# Columns: i j k m  elem_tag  type_tag  J_mean(Hz)  J_se(Hz)  phi_mean(rad)\n")
            for qid, (i, j, k, m), elem_tag, ttag, J_mean, J_se, phi_mean in results:
                f.write(f"{i:4d} {j:4d} {k:4d} {m:4d}  {elem_tag:>10s}  {ttag:>7s}  "
                        f"{J_mean:8.3f} {J_se:8.3f} {phi_mean:8.3f}\n")
        print(f"[INFO] Saved summary to {summary_file}")

    # ------------- Table-style summary (per type) -------------
    per_type_Jseries = collections.defaultdict(list)
    for J_arr, tag in zip(all_J, type_tags):
        if tag in ("HNHα", "HNCβ", "HNC'"):
            per_type_Jseries[tag].append(np.asarray(J_arr, float))

    if per_type_Jseries:
        print("\n[TABLE-STYLE SUMMARY]  mean(SE)  Hz")
        for tag in ("HNHα", "HNCβ", "HNC'"):
            if tag not in per_type_Jseries or len(per_type_Jseries[tag]) == 0:
                continue
            stack = np.stack(per_type_Jseries[tag], axis=0)  # (n_res, nF)
            J_bar_t = stack.mean(axis=0)                     # frame-wise average across residues
            J_mean  = J_bar_t.mean() if w is None else np.average(J_bar_t, weights=w)
            J_se    = block_std_err(J_bar_t, weights=w, nblock=args.nblock)
            print(f"{tag:5s}: {J_mean:.2f} ({J_se:.2f})   [n_res={stack.shape[0]}, n_frames={stack.shape[1]}]")

    # concise list
    print("\n[SUMMARY] Per-quartet results:")
    print("Quartet (elem)          type     J (Hz)       Error")
    print("-" * 62)
    for _, (i,j,k,m), elem_tag, ttag, J_mean, J_se, _ in results:
        print(f"{i}-{j}-{k}-{m} [{elem_tag:>10s}] {ttag:>7s}: {J_mean:7.2f} ± {J_se:5.2f}")

if __name__ == "__main__":
    main()
