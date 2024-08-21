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
    
    
    
    
# # 解析SMILES字符串为分子对象  
# smiles = "CC1=CC=C(N2CCCC2)C=C1C"  
# mol = Chem.MolFromSmiles(smiles)  
  
# # 获取具有最多隐式氢的原子的位置  
# positions = find_max_implicit_hydrogens_position(mol)  
# print(f"具有最多隐式氢的原子的位置: {positions}")




filename = 'frag_data.csv'  
df = pd.read_csv(filename)  
filtered_df = df[df['count'] > 10]  
filtered_frag_smiles = filtered_df['frag_smiles'].tolist() 

result = []
with open("./log/break_mols_v2.json",'r',encoding='utf-8') as f:
    result = json.load(f)

for key, value in tqdm(result.items()):
    for fragment_info in value[1]:
        fragment_info.setdefault('new_frag_smiles', [])
        new_frag_smiles = []
        for frag_smiles in filtered_frag_smiles:
            new_smiles = replace_frag_with_new_frag(key, fragment_info['frag_smiles'], frag_smiles)
            new_frag = frag_smiles
            
            if len(new_smiles) == 0:
                continue
            
            new_frag_smiles.append((new_frag, new_smiles))
            
        if len(new_frag_smiles) > 0:
            fragment_info['new_frag_smiles'] = new_frag_smiles
            
with open("./log/new_frag_smiles_v2.json",'w',encoding='utf-8') as f:
    json.dump(result, f,ensure_ascii=False)
    
