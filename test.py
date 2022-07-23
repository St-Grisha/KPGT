from dgllife.utils.featurizers import (
    ConcatFeaturizer,
    bond_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
    atomic_number_one_hot,
    atom_degree_one_hot,
    atom_formal_charge,
    atom_num_radical_electrons_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_total_num_H_one_hot,
    atom_is_chiral_center,
    atom_chirality_type_one_hot,
    atom_mass,
    one_hot_encoding,
)
from functools import partial
from rdkit import Chem

PERIODIC_TABLE = Chem.GetPeriodicTable()


def period(atom):
    atomic_num = atom.GetAtomicNum()
    if atomic_num <= 2:
        return 1
    elif atomic_num <= 10:
        return 2
    elif atomic_num <= 18:
        return 3
    elif atomic_num <= 36:
        return 4
    elif atomic_num <= 54:
        return 5
    elif atomic_num <= 86:
        return 6
    else:
        return 7


def one_hot_period(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [1, 2, 3, 4, 5, 6, 7]
    return one_hot_encoding(period(atom), allowable_set, encode_unknown)


def one_hot_outer_elecs(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [1, 2, 3, 4, 5, 6, 7, 8]
    return one_hot_encoding(
        PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum()),
        allowable_set,
        encode_unknown,
    )


def get_rvdv(atom):
    return [PERIODIC_TABLE.GetRvdw(atom.GetAtomicNum())]


def get_rcov(atom):
    return [PERIODIC_TABLE.GetRcovalent(atom.GetAtomicNum())]


bond_featurizer_all = ConcatFeaturizer(
    [  # 15
        partial(
            bond_type_one_hot,
            allowable_set=[
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
                Chem.rdchem.BondType.DATIVE,
            ],
            encode_unknown=True,
        ),  # 6
        bond_is_conjugated,  # 1
        bond_is_in_ring,  # 1
        partial(bond_stereo_one_hot, encode_unknown=True),  # 7
    ]
)
atom_featurizer_all = ConcatFeaturizer(
    [  # 53
        # partial(atomic_number_one_hot, encode_unknown=True), #101
        one_hot_period,  # 7
        partial(one_hot_outer_elecs, encode_unknown=True),  # 8
        partial(atom_degree_one_hot, encode_unknown=True),  # 12
        atom_formal_charge,  # 1
        partial(atom_num_radical_electrons_one_hot, encode_unknown=True),  # 6
        partial(atom_hybridization_one_hot, encode_unknown=True),  # 6
        atom_is_aromatic,  # 1
        partial(atom_total_num_H_one_hot, encode_unknown=True),  # 6
        atom_is_chiral_center,  # 1
        atom_chirality_type_one_hot,  # 2
        atom_mass,  # 1
        get_rvdv,  # 1
        get_rcov,  # 1
    ]
)


def smiles_to_graph(smiles):
    # Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    # Featurize Atoms
    n_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))
    print(atom_features)


if __name__ == "__main__":
    smiles_to_graph("CCCS")
