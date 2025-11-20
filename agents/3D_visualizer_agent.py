import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
import py3Dmol
from IPython.display import display
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class Molecule3DVisualizer:
    """
    Generate and visualize 3D molecular structures.
    Supports SDF, PDB, PNG exports and interactive 3D viewing.
    """
    
    def __init__(self):
        self.conformers = {}
    
    def generate_3d_structure(self, smiles: str, optimize: bool = True) -> Chem.Mol:
        """
        Generate 3D coordinates for a molecule from SMILES.
        
        Parameters:
        -----------
        smiles : molecule SMILES string
        optimize : whether to optimize geometry with UFF
        
        Returns:
        --------
        RDKit Mol object with 3D coordinates
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol_h = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        params = AllChem.ETKDG()
        params.randomSeed = 42
        
        result = AllChem.EmbedMolecule(mol_h, params)
        
        if result != 0:
            raise ValueError("Failed to generate 3D coordinates")
        
        # Optimize geometry
        if optimize:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        
        return mol_h
    
    def generate_multiple_conformers(self, smiles: str, n_conformers: int = 10) -> List[Chem.Mol]:
        """
        Generate multiple conformers for a molecule.
        
        Parameters:
        -----------
        smiles : molecule SMILES
        n_conformers : number of conformers to generate
        
        Returns:
        --------
        List of Mol objects with different conformations
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol_h = Chem.AddHs(mol)
        
        # Generate multiple conformers
        params = AllChem.ETKDG()
        params.randomSeed = 42
        
        conf_ids = AllChem.EmbedMultipleConfs(
            mol_h,
            numConfs=n_conformers,
            params=params
        )
        
        if len(conf_ids) == 0:
            raise ValueError("Failed to generate conformers")
        
        # Optimize each conformer
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=200)
        
        self.conformers[smiles] = mol_h
        return mol_h
    
    def save_as_sdf(self, mol: Chem.Mol, filename: str):
        """Save molecule to SDF file."""
        writer = Chem.SDWriter(filename)
        writer.write(mol)
        writer.close()
        print(f"Saved to {filename}")
    
    def save_as_pdb(self, mol: Chem.Mol, filename: str, conf_id: int = -1):
        """Save molecule to PDB file."""
        Chem.MolToPDBFile(mol, filename, confId=conf_id)
        print(f"Saved to {filename}")
    
    def save_as_png(self, smiles: str, filename: str, size=(400, 400)):
        """Save 2D structure as PNG image."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        img = Draw.MolToImage(mol, size=size)
        img.save(filename)
        print(f"Saved to {filename}")
    
    def save_3d_as_png(self, mol: Chem.Mol, filename: str, size=(400, 400)):
        """Save 3D structure as PNG image (multiple views)."""
        # Generate 2D depiction with 3D-like perspective
        img = Draw.MolToImage(mol, size=size)
        img.save(filename)
        print(f"Saved to {filename}")
    
    def view_3d_interactive(self, mol: Chem.Mol, conf_id: int = -1, 
                           style: str = 'stick', color_scheme: str = 'default'):
        """
        Display interactive 3D visualization in Jupyter notebook.
        
        Parameters:
        -----------
        mol : RDKit Mol object with 3D coordinates
        conf_id : conformer ID to display (-1 for first/only conformer)
        style : 'stick', 'sphere', 'cartoon', 'line'
        color_scheme : 'default', 'greenCarbon', 'cyanCarbon', etc.
        
        Returns:
        --------
        py3Dmol view object
        """
        # Convert to PDB format
        pdb = Chem.MolToPDBBlock(mol, confId=conf_id)
        
        # Create viewer
        viewer = py3Dmol.view(width=800, height=600)
        viewer.addModel(pdb, 'pdb')
        
        # Set style
        if style == 'stick':
            viewer.setStyle({'stick': {'colorscheme': color_scheme}})
        elif style == 'sphere':
            viewer.setStyle({'sphere': {'colorscheme': color_scheme}})
        elif style == 'cartoon':
            viewer.setStyle({'cartoon': {'color': 'spectrum'}})
        elif style == 'line':
            viewer.setStyle({'line': {'colorscheme': color_scheme}})
        
        # Add surface (optional)
        # viewer.addSurface(py3Dmol.VDW, {'opacity': 0.7})
        
        viewer.zoomTo()
        viewer.show()
        
        return viewer
    
    def view_3d_from_smiles(self, smiles: str, style: str = 'stick'):
        """Quick 3D view directly from SMILES."""
        mol_3d = self.generate_3d_structure(smiles)
        return self.view_3d_interactive(mol_3d, style=style)
    
    def compare_conformers(self, mol: Chem.Mol, max_display: int = 5):
        """
        Display multiple conformers side by side.
        
        Parameters:
        -----------
        mol : Mol object with multiple conformers
        max_display : maximum number of conformers to display
        """
        n_confs = mol.GetNumConformers()
        n_display = min(n_confs, max_display)
        
        viewers = []
        for i in range(n_display):
            pdb = Chem.MolToPDBBlock(mol, confId=i)
            viewer = py3Dmol.view(width=300, height=300)
            viewer.addModel(pdb, 'pdb')
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer.show()
            viewers.append(viewer)
            print(f"Conformer {i + 1}")
        
        return viewers
    
    def get_molecular_properties(self, mol: Chem.Mol) -> dict:
        """Calculate 3D molecular properties."""
        props = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        }
        
        # 3D properties (if 3D coordinates exist)
        if mol.GetNumConformers() > 0:
            from rdkit.Chem import rdMolDescriptors
            props.update({
                'asphericity': rdMolDescriptors.CalcAsphericity(mol),
                'eccentricity': rdMolDescriptors.CalcEccentricity(mol),
                'radius_of_gyration': rdMolDescriptors.CalcRadiusOfGyration(mol),
                'spherocity_index': rdMolDescriptors.CalcSpherocityIndex(mol),
            })
        
        return props
    
    def batch_generate_3d(self, smiles_list: List[str], output_dir: str = '.'):
        """
        Generate 3D structures for multiple molecules and save to files.
        
        Parameters:
        -----------
        smiles_list : list of SMILES strings
        output_dir : directory to save files
        """
        results = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                print(f"Processing molecule {i + 1}/{len(smiles_list)}...")
                
                # Generate 3D
                mol_3d = self.generate_3d_structure(smiles)
                
                # Save files
                base_name = f"{output_dir}/molecule_{i + 1}"
                self.save_as_sdf(mol_3d, f"{base_name}.sdf")
                self.save_as_pdb(mol_3d, f"{base_name}.pdb")
                self.save_as_png(smiles, f"{base_name}_2d.png")
                
                # Get properties
                props = self.get_molecular_properties(mol_3d)
                props['smiles'] = smiles
                props['file_prefix'] = base_name
                results.append(props)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Save summary
        import pandas as pd
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(f"{output_dir}/3d_summary.csv", index=False)
        
        print(f"\nGenerated 3D structures for {len(results)} molecules")
        return df_summary
    
    def load_from_sdf(self, sdf_file: str) -> List[Chem.Mol]:
        """Load molecules from SDF file (e.g., your AfroDB)."""
        suppl = Chem.SDMolSupplier(sdf_file)
        mols = [mol for mol in suppl if mol is not None]
        print(f"Loaded {len(mols)} molecules from {sdf_file}")
        return mols
    
    def load_from_pdb(self, pdb_file: str) -> Chem.Mol:
        """Load molecule from PDB file (e.g., your protein targets)."""
        mol = Chem.MolFromPDBFile(pdb_file)
        if mol is None:
            raise ValueError(f"Failed to load {pdb_file}")
        print(f"Loaded molecule from {pdb_file}")
        return mol
    
    def visualize_protein_ligand(self, protein_pdb: str, ligand_smiles: str):
        """
        Visualize protein-ligand complex.
        
        Parameters:
        -----------
        protein_pdb : path to protein PDB file
        ligand_smiles : ligand SMILES string
        """
        # Load protein
        with open(protein_pdb, 'r') as f:
            protein_pdb_str = f.read()
        
        # Generate ligand 3D
        ligand_3d = self.generate_3d_structure(ligand_smiles)
        ligand_pdb = Chem.MolToPDBBlock(ligand_3d)
        
        # Create viewer
        viewer = py3Dmol.view(width=800, height=600)
        
        # Add protein
        viewer.addModel(protein_pdb_str, 'pdb')
        viewer.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})
        
        # Add ligand
        viewer.addModel(ligand_pdb, 'pdb')
        viewer.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
        
        viewer.zoomTo()
        viewer.show()
        
        return viewer

'''
# USAGE EXAMPLES:
"""
# Initialize visualizer
viz = Molecule3DVisualizer()

# Example 1: Generate and view single molecule
smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
mol_3d = viz.generate_3d_structure(smiles)

# Save to different formats
viz.save_as_sdf(mol_3d, 'ibuprofen.sdf')
viz.save_as_pdb(mol_3d, 'ibuprofen.pdb')
viz.save_as_png(smiles, 'ibuprofen_2d.png')

# Interactive 3D view (in Jupyter)
viz.view_3d_interactive(mol_3d, style='stick')

# Example 2: Generate multiple conformers
mol_confs = viz.generate_multiple_conformers(smiles, n_conformers=10)
viz.compare_conformers(mol_confs, max_display=5)

# Example 3: Batch process your optimized molecules
optimized_smiles = ['SMILES1', 'SMILES2', 'SMILES3']
summary = viz.batch_generate_3d(optimized_smiles, output_dir='./3d_structures')

# Example 4: Load and visualize your AfroDB
afrodb_mols = viz.load_from_sdf('your_afrodb.sdf')
for mol in afrodb_mols[:5]:  # First 5
    viz.view_3d_interactive(mol)

# Example 5: Visualize with protein target
viz.visualize_protein_ligand(
    protein_pdb='your_target_protein.pdb',
    ligand_smiles='CC(C)Cc1ccc(cc1)C(C)C(=O)O'
)

# Example 6: Quick view from SMILES
viz.view_3d_from_smiles('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin'''
"""
