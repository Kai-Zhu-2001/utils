import numpy as np
from ase.data import covalent_radii as CR
from ase.data import chemical_symbols as CS
from ase.io import read
from ase import Atoms
import pandas as pd
from mlcolvar.utils.io import load_dataframe


def create_chemiscope_input(
    trajectory, filename=None, colvar=None, cvs=["*"], verbose=False
):
    """
    Create a chemiscope input file from a trajectory and optional collective variables (colvar) file.

    Parameters
    ----------
    trajectory : list of ase.Atoms or str
        Trajectory of atoms objects or path to an xyz file
    filename : str, optional
        Output filename. If None, it will be saved with the same name of the trajectory with _chemiscope.json.gz appended.
    colvar : str or pandas.DataFrame, optional
        Path of the COLVAR file or a pandas dataframe
    cvs : list of str, optional
        List of collective variable names to be saved into the chemiscope file. If a string contains '*', it will be used as a filter for the property names in the colvar file (e.g. 'cv.*' will extract all properties with 'cv.' prefix). Default is ['*']
    verbose: bool, optional
        Print information
    Returns
    -------
    filename : str
        Path of the chemiscope input file
    """

    if verbose:
        print("[INFO] Creating Chemiscope input file...")
    try:
        import chemiscope
    except ImportError:
        raise ImportError(
            "Chemiscope is not installed. Please install it with pip install chemiscope"
        )

    # check if trajectory is a list of atoms or a filename
    if isinstance(trajectory, list) & isinstance(trajectory[0], Atoms):
        traj = trajectory
    elif isinstance(trajectory, str):
        if verbose:
            print("[INFO] Reading file:", trajectory)
        traj = read(trajectory, index=":")
    atoms = traj[0]

    # load colvar file into traj if requested
    if colvar is not None:
        if isinstance(colvar, pd.DataFrame):
            pass
        else:
            try:
                colvar = load_dataframe(colvar)

            except Exception as e:
                print(
                    f"[WARNING]: colvar file: {colvar} not read, it should be a string filename or a pandas dataframe. Exception: {e}."
                )
            
        try:
        # if time in colvar use colvar.time and atoms.info['frame'] to ensure consistency
            dt, consistent = None, True
            atoms_old = traj[0]
            if "time" in colvar.columns and "frame" in atoms_old.info:
                # check if time is consistent between colvar and traj
                if verbose:
                    print(
                        f"[INFO] Checking time consistency between COLVAR and trajectory..."
                    )
                for i, atoms in enumerate(traj[1:]):
                    if dt is None:
                        frames = atoms.info["frame"] - atoms_old.info["frame"]
                        time_interval = (
                            colvar["time"].loc[atoms.info["frame"]]
                            - colvar["time"].loc[atoms_old.info["frame"]]
                        )
                        dt = time_interval / frames
                    else:  # check consistency
                        frames = atoms.info["frame"] - atoms_old.info["frame"]
                        time_interval = (
                            colvar["time"].loc[atoms.info["frame"]]
                            - colvar["time"].loc[atoms_old.info["frame"]]
                        )
                        dt_new = time_interval / frames
                        if np.abs(dt - dt_new) > 1e-8:
                            consistent = False
                    atoms_old = atoms
                if consistent:
                    if verbose:
                        print(
                            f"[INFO] Time consistency between COLVAR and trajectory verified."
                        )
                    for i, atoms in enumerate(traj):
                        for col in colvar.columns:
                            atoms.info["colvar." + col] = colvar[col].loc[
                                atoms.info["frame"]
                            ]
                else:
                    print(
                        f"[WARNING]: time inconsistency between COLVAR and trajectory detected. Not saving COLVAR information."
                    )
                    for i, atoms in enumerate(traj):
                        for col in colvar.columns:
                            atoms.info["colvar." + col] = colvar[col].iloc[i]
            elif len(colvar) == len(traj):
                print(
                    "[WARNING]: Consistency between traj and COLVAR cannot be assessed. Saving COLVAR assuming that the order of frames in COLVAR and trajectory are the same."
                )
                for i, atoms in enumerate(traj):
                    for col in colvar.columns:
                        atoms.info["colvar." + col] = colvar[col].iloc[i]
            else:
                print(
                    "[WARNING]: Consistency between traj and COLVAR cannot be assessed and lengths do not match. Not saving COLVAR information."
                )
        except Exception as e:
            print(e)

    # Get CV names
    prop_names, prop_names_float = [], []
    for c in cvs:
        if "*" in c:
            prop_names.extend([p for p in atoms.info.keys() if c.replace("*", "") in p])
        else:
            prop_names.append(c)

    # Check if CV names can be converted to float
    for p in prop_names:
        if p == "target_atoms":
            continue
        try:
            float(atoms.info[p])
            prop_names_float.append(p)
        except TypeError:
            # if p != "target_atoms":
            print(f'skipping "{p}" as it cannot be converted to float.')

    if verbose:
        print("[INFO] CV names:", prop_names_float)

    # Extract properties
    properties = chemiscope.extract_properties(traj, only=prop_names_float)

    # Define shape and colors
    shapes_selection = []
    for atoms in traj:
        target_atoms = atoms.info.get("target_atoms", None)
        for i, atom in enumerate(atoms):
            if target_atoms is not None:
                if not isinstance(target_atoms, np.ndarray):
                    target_atoms = np.asarray([target_atoms])
                shapes_selection.append(
                    {
                        "radius": CR[CS.index(atom.symbol)],
                        "color": None if i in target_atoms else "#d4d4d4",
                    }
                )
            else:
                shapes_selection.append(
                    {"radius": CR[CS.index(atom.symbol)], "color": None}
                )
    if target_atoms is not None and verbose:
        print('[INFO] "target_atoms" found in atoms.')

    # Write input
    if filename is None:
        if isinstance(trajectory, str):
            filename = os.path.splitext(trajectory)[0] + "_chemiscope.json.gz"
        else:
            filename = "chemiscope.json.gz"

    chemiscope.write_input(
        filename,
        frames=traj,
        properties=properties,
        meta=dict(name="DEAL selection"),
        shapes={
            "selection": {"kind": "sphere", "parameters": {"atom": shapes_selection}},
        },
        settings={
            "structure": [
                {
                    "atoms": False,
                    "bonds": False,
                    "shape": "selection",
                    "axes": "off",
                    "keepOrientation": False,
                    "playbackDelay": 700,
                }
            ]
        },
    )

    if verbose:
        print("[OUTPUT] Chemiscope input saved in:", filename)

    return filename
