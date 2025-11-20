import subprocess
import tempfile
import os
import re # Not strictly needed, but good practice for robust parsing
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy

class SimpleDockingAgent:
    """
    Minimal AutoDock Vina wrapper with robust error reporting.
    """
    
    def __init__(self, protein_pdbqt, binding_site):
        """
        Parameters:
        -----------
        protein_pdbqt : str
            Path to prepared protein PDBQT file
        binding_site : dict
            {'center': (x, y, z), 'size': (x, y, z)}
        """
        self.protein_pdbqt = protein_pdbqt
        self.center = binding_site['center']
        self.size = binding_site['size']
    
    def smiles_to_3d(self, smiles):
        """
        Convert SMILES to 3D molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            params = AllChem.ETKDG()
            params.randomSeed = 42
            
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None
            
            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol)
            
            return mol
            
        except Exception as e:
            print(f"3D generation failed: {e}")
            return None
    
    def mol_to_pdbqt(self, mol):
        """
        Convert RDKit molecule to PDBQT string using Meeko
        """
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        # Note: error_msg is not used here but is returned by Meeko
        pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        
        return pdbqt_string if is_ok else None

    def run_vina(self, ligand_pdbqt_content):
        """
        Run AutoDock Vina docking with built-in error reporting.
        (FIXED: Removed unsupported '--log' and parse stdout instead)
        """
        ligand_file, output_file = None, None
        
        try:
            # 1. Write ligand PDBQT to temp file
            with tempfile.NamedTemporaryFile(suffix='_ligand.pdbqt', delete=False, mode='w') as f:
                f.write(ligand_pdbqt_content)
                ligand_file = f.name
            
            # 2. Define output files
            output_file = ligand_file.replace('_ligand.pdbqt', '_out.pdbqt')
            
            # 3. Build Vina command (Removed '--log' and log_file)
            cmd = [
                'vina',
                '--receptor', self.protein_pdbqt,
                '--ligand', ligand_file,
                '--out', output_file,
                '--center_x', str(self.center[0]),
                '--center_y', str(self.center[1]),
                '--center_z', str(self.center[2]),
                '--size_x', str(self.size[0]),
                '--size_y', str(self.size[1]),
                '--size_z', str(self.size[2]),
                '--exhaustiveness', '8',
                '--num_modes', '1',
                '--energy_range', '3'
            ]
            
            # 4. Run Vina
            # Capture stdout to get the scores (since --log is not supported)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # --- CRITICAL ERROR CHECK ---
            if result.returncode != 0:
                print(f"\n--- VINA EXECUTION FAILED! Return Code: {result.returncode} ---")
                print("Command: " + " ".join(cmd))
                print("\n--- VINA STDERR (Error Output) ---")
                print(result.stderr)
                return None
            
            # 5. Parse binding energy from STDOUT
            binding_energy = None
            
            # Vina prints the result table to stdout. We look for the first line starting with '1'.
            output_lines = result.stdout.splitlines()
            
            for line in output_lines:
                # The line format is: 1 | -8.5 | 0.000 / 0.000
                if line.strip().startswith('1'): 
                    # Use a split on whitespace to extract columns
                    parts = line.split()
                    
                    # Check if the first part is exactly '1' and there's a second part (the score)
                    if len(parts) >= 2 and parts[0] == '1':
                        try:
                            # The binding energy is the second column
                            binding_energy = float(parts[1])
                            break
                        except ValueError:
                            # Handle cases where the second part might not be a float
                            continue 
            
            return binding_energy
            
        except subprocess.TimeoutExpired:
            print("Vina timeout (>2 min)")
            return None
        except Exception as e:
            print(f"Vina execution failed: {e}") 
            return None
        finally:
            # 6. Cleanup (only ligand and output pdbqt files)
            for f in [ligand_file, output_file]:
                if f and os.path.exists(f):
                    os.remove(f)
    
    def dock_compound(self, smiles):
        """
        Complete docking workflow for a SMILES string
        """
        try:
            # Step 1: Convert SMILES to 3D
            mol = self.smiles_to_3d(smiles)
            if mol is None:
                return None
            
            # Step 2: Convert to PDBQT format
            pdbqt_content = self.mol_to_pdbqt(mol)
            if pdbqt_content is None:
                return None
            
            # Step 3: Run Vina docking 
            binding_energy = self.run_vina(pdbqt_content)
            
            return binding_energy
            
        except Exception as e:
            print(f"Docking failed for {smiles[:20]}...: {e}")
            return None
    
    def dock_batch(self, smiles_list, compound_names=None):
        """
        Dock multiple compounds
        """
        results = []
        
        for idx, smiles in enumerate(smiles_list):
            name = compound_names[idx] if compound_names else f"Compound_{idx+1}"
            
            print(f"Docking {name} ({idx+1}/{len(smiles_list)})...", end=' ')
            
            energy = self.dock_compound(smiles)
            
            if energy is not None:
                print(f"✅ {energy:.2f} kcal/mol")
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'binding_energy': energy,
                    'status': 'Success'
                })
            else:
                print(f"❌ Failed")
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'binding_energy': None,
                    'status': 'Failed'
                })
        
        return pd.DataFrame(results)
def prepare_protein_for_docking(pdb_file, output_pdbqt):
    """Prepare protein using Open Babel"""
    try:
        cmd = ['obabel', pdb_file, '-O', output_pdbqt, '-xr', '-p', '7.4']
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False



    DOCKING_TARGETS = {
    # CANCER TARGETS
    'cancer_EGFR': {
        'pdb_id': '1M17',
        'binding_site': {'center': (25.0, 12.5, 40.3), 'size': (20, 20, 20)},
        'models': ['cancer_EGFR_regression', 'cancer_EGFR_classification']
    },
    'cancer_BCR_ABL': {
        'pdb_id': '2HYY',
        'binding_site': {'center': (15.3, 22.1, 18.7), 'size': (20, 20, 20)},
        'models': ['cancer_BCR_ABL_regression', 'cancer_BCR_ABL_classification']
    },
    'cancer_HER2': {
        'pdb_id': '3PP0',
        'binding_site': {'center': (18.5, 14.2, 32.1), 'size': (20, 20, 20)},
        'models': ['cancer_HER2_regression', 'cancer_HER2_classification']
    },
    
    # HIV TARGETS
    'hiv_RT': {
        'pdb_id': '1RTD',
        'binding_site': {'center': (15.2, 18.7, 22.1), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_RT_regression', 'hiv_HIV_RT_classification']
    },
    'hiv_Protease': {
        'pdb_id': '1HXB',
        'binding_site': {'center': (0.5, 1.2, -2.3), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_Protease_regression', 'hiv_HIV_Protease_classification']
    },
    'hiv_Integrase': {
        'pdb_id': '1QS4',
        'binding_site': {'center': (12.3, 8.5, 15.2), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_Integrase_regression', 'hiv_HIV_Integrase_classification']
    },
    
    # DIABETES TARGETS
    'diabetes_DPP4': {
        'pdb_id': '1X70',
        'binding_site': {'center': (25.1, 15.3, 10.8), 'size': (20, 20, 20)},
        'models': ['diabetes_DPP4_regression', 'diabetes_DPP4_classification']
    },
    
    # HYPERTENSION
    'hypertension_ACE': {
        'pdb_id': '1O86',
        'binding_site': {'center': (30.2, 28.5, 42.1), 'size': (20, 20, 20)},
        'models': ['hypertension_ACE_regression', 'hypertension_ACE_classification']
    },
    
    # INFLAMMATION
    'inflammation_COX2': {
        'pdb_id': '5KIR',
        'binding_site': {'center': (28.5, 22.3, 15.8), 'size': (20, 20, 20)},
        'models': ['inflammation_COX2_regression', 'inflammation_COX2_classification']
    }
}
DOCKING_TARGETS.update({
    'cancer_CDK': {
        'pdb_id': '1HCK',
        'binding_site': {'center': (15.5, 22.3, 18.8), 'size': (20, 20, 20)},
        'models': ['cancer_CDK_regression', 'cancer_CDK_classification']
    },
    'tuberculosis': {
        'pdb_id': '4TZK',
        'binding_site': {'center': (12.8, 18.5, 22.3), 'size': (20, 20, 20)},
        'models': ['tuberculosis_classification']
    }
})

# Disease to target mapping
DISEASE_TARGET_MAP = {
    'cancer': ['cancer_EGFR', 'cancer_BCR_ABL', 'cancer_HER2', 'cancer_VEGFR2', 'cancer_Topo_II', 'cancer_CDK'],
    'hiv': ['hiv_RT', 'hiv_Protease', 'hiv_Integrase'],
    'diabetes': ['diabetes_DPP4', 'diabetes_Alpha_Glucosidase', 'diabetes_PPAR_gamma'],
    'hypertension': ['hypertension_ACE'],
    'inflammation': ['inflammation_COX2', 'inflammation_LOX5'],
    'tuberculosis': ['tuberculosis']
}

# Add to DOCKING_TARGETS dictionary (append, don't replace)

DOCKING_TARGETS.update({
    'inflammation_LOX5': {
        'pdb_id': '3O8Y',
        'binding_site': {'center': (20.5, 15.3, 25.1), 'size': (20, 20, 20)},
        'models': ['inflammation_LOX5_regression', 'inflammation_LOX5_classification']
    },
    'tuberculosis_InhA': {
        'pdb_id': '4TZK',
        'binding_site': {'center': (12.8, 18.5, 22.3), 'size': (20, 20, 20)},
        'models': ['tuberculosis_InhA_regression', 'tuberculosis_classification']
    }
})

# Add to disease map
DISEASE_TARGET_MAP.update({
    'tuberculosis': ['tuberculosis_InhA']
})

# Update existing inflammation
DISEASE_TARGET_MAP['inflammation'].append('inflammation_LOX5')
# Add these to DOCKING_TARGETS

DOCKING_TARGETS.update({
    'cancer_VEGFR2': {
        'pdb_id': '3WZE',
        'binding_site': {'center': (30.2, 18.5, 22.7), 'size': (20, 20, 20)},
        'models': ['cancer_VEGFR2_regression', 'cancer_VEGFR2_classification']
    },
    'cancer_Topo_II': {
        'pdb_id': '3QX3',
        'binding_site': {'center': (25.8, 32.1, 15.5), 'size': (20, 20, 20)},
        'models': ['cancer_Topo_II_classification']
    },
    'diabetes_Alpha_Glucosidase': {
        'pdb_id': '3L4Y',
        'binding_site': {'center': (22.5, 28.3, 35.2), 'size': (20, 20, 20)},
        'models': ['diabetes_Alpha_Glucosidase_regression', 'diabetes_Alpha_Glucosidase_classification']
    },
    'diabetes_PPAR_gamma': {
        'pdb_id': '2PRG',
        'binding_site': {'center': (18.3, 12.7, 20.8), 'size': (20, 20, 20)},
        'models': ['diabetes_PPAR_gamma_classification']
    }
})

# Update disease maps
DISEASE_TARGET_MAP['cancer'].extend(['cancer_VEGFR2', 'cancer_Topo_II'])
DISEASE_TARGET_MAP['diabetes'].extend(['diabetes_Alpha_Glucosidase', 'diabetes_PPAR_gamma'])

def prepare_protein_for_docking(pdb_file, output_pdbqt):
    """Prepare protein using obabel"""
    try:
        cmd = ['obabel', pdb_file, '-O', output_pdbqt, '-xr', '-p', '7.4']
        result = subprocess.run(cmd, capture_output=True, check=True)
        print(f"✅ {output_pdbqt}")
        return True
    except Exception as e:
        print(f"❌ {output_pdbqt}: {e}")
        return False

# 3. Re-run setup (will now actually create files)
def setup_all_targets_rapid():
    """Setup all docking targets - FIXED"""
    import subprocess
    
    dockers = {}
    
    for target_name, info in DOCKING_TARGETS.items():
        print(f"⚙️ {target_name}...", end=' ')
        
        pdb_file = f'{target_name}.pdb'
        pdbqt_file = f'{target_name}.pdbqt'
        
        # Download (only if not exists)
        if not os.path.exists(pdb_file):
            !wget -q https://files.rcsb.org/download/{info['pdb_id']}.pdb -O {pdb_file}
        
        # Prepare protein
        if prepare_protein_for_docking(pdb_file, pdbqt_file):
            # Verify file exists and has content
            if os.path.exists(pdbqt_file) and os.path.getsize(pdbqt_file) > 100:
                dockers[target_name] = SimpleDockingAgent(pdbqt_file, info['binding_site'])
                print(f"✅ ({os.path.getsize(pdbqt_file)} bytes)")
            else:
                print("❌ File empty/missing")
        else:
            print("❌ Prep failed")
    
    print(f"\n✅ Setup complete: {len(dockers)}/{len(DOCKING_TARGETS)} ready")
    return dockers

import pandas as pd
import os

# Assume all_dockers dictionary is populated by setup_all_targets_rapid()
# If you run this code separately, you must ensure all_dockers is defined and populated.

def perform_docking_for_target(smiles, target_name, debug=False):
    """
    Executes a molecular docking simulation for a single compound against a specific target.

    Parameters:
    -----------
    smiles : str
        The SMILES string of the compound to dock.
    target_name : str
        The name of the target protein (e.g., 'cancer_EGFR').
    debug : bool
        If True, prints detailed debug messages during the workflow.

    Returns:
    --------
    dict: {'target': str, 'smiles': str, 'binding_energy': float or None, 'status': str}
    """
    if target_name not in all_dockers:
        return {
            'target': target_name, 
            'smiles': smiles, 
            'binding_energy': None, 
            'status': f"Error: Target '{target_name}' not found/prepared."
        }
    
    # 1. Get the pre-configured SimpleDockingAgent
    docker = all_dockers[target_name]
    
    if debug:
        print(f"DEBUG: Starting docking for SMILES: {smiles} on target: {target_name}")

    # 2. Execute the docking workflow
    energy = docker.dock_compound(smiles)

    if energy is not None:
        status = 'Success'
        if debug:
            print(f"DEBUG: Docking successful. Energy: {energy:.2f} kcal/mol.")
    else:
        status = 'Failed'
        if debug:
            # Note: The underlying SimpleDockingAgent handles printing specific failure reasons
            print(f"DEBUG: Docking failed.")

    # 3. Return a structured result
    return {
        'target': target_name,
        'smiles': smiles,
        'binding_energy': energy,
        'status': status
    }

# ============= ADD THIS NEW SECTION =============
def save_docking_results_to_csv(all_dockers, compounds_df, output_csv='docking_results.csv'):
    """
    Dock all compounds against all targets and save comprehensive results
    
    Parameters:
    -----------
    all_dockers : dict
        Dictionary of SimpleDockingAgent instances {target_name: docker}
    compounds_df : DataFrame
        Must have 'canonical_smiles' and optionally 'compound_name' columns
    output_csv : str
        Output filename
    """
    import pandas as pd
    from tqdm import tqdm
    
    all_results = []
    
    total_dockings = len(compounds_df) * len(all_dockers)
    print(f"🎯 Starting {total_dockings} docking simulations...")
    print(f"   {len(compounds_df)} compounds × {len(all_dockers)} targets\n")
    
    # For each target
    for target_name, docker in all_dockers.items():
        print(f"\n{'='*70}")
        print(f"🎯 Docking against: {target_name.upper()}")
        print(f"{'='*70}")
        
        target_results = []
        
        # Dock all compounds
        for idx, row in tqdm(compounds_df.iterrows(), total=len(compounds_df), desc=f"{target_name}"):
            smiles = row['canonical_smiles']
            compound_name = row.get('compound_name', f"Compound_{idx}")
            
            # Perform docking
            energy = docker.dock_compound(smiles)
            
            result = {
                'compound_name': compound_name,
                'smiles': smiles,
                'target': target_name,
                'binding_energy': energy,
                'binding_affinity': energy,  # Same as binding_energy
                'target_protein': target_name.split('_')[0],  # e.g., 'cancer' from 'cancer_EGFR'
                'status': 'Success' if energy is not None else 'Failed'
            }
            
            target_results.append(result)
            all_results.append(result)
        
        # Summary for this target
        successes = sum(1 for r in target_results if r['status'] == 'Success')
        print(f"✅ {target_name}: {successes}/{len(compounds_df)} successful")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"📊 DOCKING SUMMARY")
    print(f"{'='*70}")
    print(f"Total simulations: {len(results_df)}")
    print(f"Successful: {(results_df['status'] == 'Success').sum()}")
    print(f"Failed: {(results_df['status'] == 'Failed').sum()}")
    print(f"\n💾 Saved to: {output_csv}")
    
    return results_df

