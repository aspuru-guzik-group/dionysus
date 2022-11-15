from __future__ import annotations

from typing import Optional, List

import numpy as np
import rdkit
import rdkit.Chem.AllChem as Chem

from . import types


def smi_to_mol(smi: str) -> Optional[types.Mol]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol
    except ValueError:
        return None


def get_isosmiles(mol: types.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def get_mfp(mol: types.Mol) -> np.ndarray:
    return np.array(Chem.GetMorganFingerprintAsBitVect(mol, 3))


def manysmi_to_fps(smi_arr: Sequence[str]) -> np.ndarray:
    return np.array([get_mfp(smi_to_mol(s)) for s in smi_arr])


def is_single(mol: types.Mol) -> bool:
    smi = get_isosmiles(mol)
    return len(Chem.GetMolFrags(mol)) == 1 or not ('.' in smi)


def is_larger_molecule(mol: types.Mol) -> bool:
    return mol.GetNumAtoms() > 1


def get_atom_set(mols: List[types.Mol], add_null: bool = True) -> List[str]:
    """Get atom set for a list of smiles."""
    atom_set = set()
    for m in mols:
        atom_set.update(set([a.GetSymbol() for a in m.GetAtoms()]))
    if add_null:
        atom_set.add('*')
    table = rdkit.Chem.GetPeriodicTable()
    atom_num = [table.GetAtomicNumber(a) for a in atom_set]
    atom_set = [a for _, a in sorted(zip(atom_num, atom_set))]
    return atom_set
