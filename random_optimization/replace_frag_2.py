import json
import pandas as pd  

from rdkit import Chem  
from rdkit.Chem import AllChem
  
from tqdm import tqdm  

import rdkit
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
  
def get_implicit_hydrogens_with_positions(mol, element):  
    # 遍历分子中的所有原子，并跟踪位置  
    for index, atom in enumerate(mol.GetAtoms()):  
        if atom.GetSymbol() == element:  
            # valence = atom.GetTotalValence()  
            # num_bonds = atom.GetDegree()  
            # implicit_hydrogens = valence - num_bonds  
            implicit_hydrogens = atom.GetTotalNumHs()
            if implicit_hydrogens > 0:  
                yield (implicit_hydrogens, index)  
  
def find_max_implicit_hydrogens_position(mol, elements=['C', 'N', 'O', 'S', 'P']):  
    max_hydrogens = 0  
    max_positions = []  
      
    for element in elements:  
        for implicit_hydrogens, position in get_implicit_hydrogens_with_positions(mol, element):  
            if implicit_hydrogens > max_hydrogens:  
                max_hydrogens = implicit_hydrogens  
                max_positions = [position]  
            elif implicit_hydrogens == max_hydrogens:  
                max_positions.append(position)  
      
    if not max_positions:  
        return [0]  # 如果没有找到任何隐式氢，则返回 [0]  
      
    return max_positions  

def replace_frag_with_new_frag(smiles, frag_smiles, new_frag_smiles):  
    m = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(frag_smiles)
    repl = Chem.MolFromSmiles(new_frag_smiles)
    replacementConnectionPoint = find_max_implicit_hydrogens_position(m)
    
    res = []
    
    try:
        for place in replacementConnectionPoint:
            rms = AllChem.ReplaceSubstructs(m, patt, repl, replacementConnectionPoint=place)
            for r in rms:
                r_str = Chem.MolToSmiles(r)
                if r_str.count('.') > 0 or Chem.MolFromSmiles(r_str) is None or r_str in res:
                    continue
                res.append(r_str)
    except:
        return res
            
    return res
    

def modify_molecule(smiles, atoms_to_remove, fragment_smiles, fragment_smiles_idx=0):
    # 将 SMILES 转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # 使用 EditableMol 进行编辑
    emol = Chem.EditableMol(mol)

    # 记录要删除的原子索引及其邻居原子索引
    neighbors = {}
    for idx in atoms_to_remove:
        atom = mol.GetAtomWithIdx(idx)
        neighbors[idx] = [nbr.GetIdx() for nbr in atom.GetNeighbors()]

    # 删除指定位置的原子
    atoms_to_remove = sorted(atoms_to_remove, reverse=True)  # 从高到低排序，避免索引问题
    for idx in atoms_to_remove:
        emol.RemoveAtom(idx)

    # 获取编辑后的分子对象
    mol = emol.GetMol()

    # 添加新的片段
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        raise ValueError("Invalid fragment SMILES string")

    # 将分子和片段合并
    combined = Chem.CombineMols(mol, fragment)

    # 使用 EditableMol 添加新的键
    emol_combined = Chem.EditableMol(combined)

    # 获取片段的连接原子索引
    fragment_atom_idx = mol.GetNumAtoms()  # 片段的第一个原子在组合分子中的位置

    # 连接到最后一个被删除的原子的第一个邻居
    mol_atom_idx = neighbors[atoms_to_remove[-1]][0] - 1

    # 添加单键
    emol_combined.AddBond(mol_atom_idx, fragment_atom_idx + fragment_smiles_idx, order=Chem.BondType.SINGLE)

    # 获取最终分子
    result = emol_combined.GetMol()

    # 更新分子的拓扑结构
    AllChem.SanitizeMol(result)

    return result
    
    
# # 解析SMILES字符串为分子对象  
# smiles = "CC1=CC=C(N2CCCC2)C=C1C"  
# mol = Chem.MolFromSmiles(smiles)  
  
# # 获取具有最多隐式氢的原子的位置  
# positions = find_max_implicit_hydrogens_position(mol)  
# print(f"具有最多隐式氢的原子的位置: {positions}")




filename = '/home/lk/project/repaint/MolPaint/random_optimization/frag_data.csv'  
df = pd.read_csv(filename)  
filtered_df = df[df['count'] > 10]  
filtered_frag_smiles = filtered_df['frag_smiles'].tolist() 

result = []
with open("/home/lk/project/repaint/MolPaint/log/break_mols_dict_20_30.json",'r',encoding='utf-8') as f:
    result = json.load(f)

for key, value in tqdm(result.items()):
    for fragment_info in value[1]:
        # import pdb;pdb.set_trace()
        fragment_info.setdefault('new_frag_smiles', [])
        new_frag_smiles = []
        for frag_smiles in filtered_frag_smiles:
            new_frag = frag_smiles
            
            new_smiles = set()
            replacementConnectionPoint = find_max_implicit_hydrogens_position(Chem.MolFromSmiles(new_frag))
            for place in replacementConnectionPoint:
                try:
                    new_smiles.add(Chem.MolToSmiles(modify_molecule(key, fragment_info['frag_id'], frag_smiles, place)))
                except:
                    continue
            if len(new_smiles) == 0:
                continue
            
            new_frag_smiles.append((new_frag, list(new_smiles)))
            
        if len(new_frag_smiles) > 0:
            fragment_info['new_frag_smiles'] = new_frag_smiles
    # 

    
with open("/home/lk/project/repaint/MolPaint/log/new_frag_smiles_20_30.json",'w',encoding='utf-8') as f:
    json.dump(result, f,ensure_ascii=False)
    
