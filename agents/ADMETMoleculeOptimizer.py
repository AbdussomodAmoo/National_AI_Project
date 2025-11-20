# dependencies
import os
import glob
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# === Paste your featurizer here (use exactly what you provided) ===
from rdkit.Chem import rdMolDescriptors
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    feature_dict = {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
    }

    # 3D embedding (best-effort)
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_3d, params) == 0:
            AllChem.UFFOptimizeMolecule(mol_3d)
            feature_dict.update({
                'Asphericity': rdMolDescriptors.CalcAsphericity(mol_3d),
                'Eccentricity': rdMolDescriptors.CalcEccentricity(mol_3d),
                'InertialShapeFactor': rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
                'RadiusOfGyration': rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
                'SpherocityIndex': rdMolDescriptors.CalcSpherocityIndex(mol_3d),
                'PMI1': rdMolDescriptors.CalcPMI1(mol_3d),
                'PMI2': rdMolDescriptors.CalcPMI2(mol_3d),
                'PMI3': rdMolDescriptors.CalcPMI3(mol_3d),
            })
        else:
            raise Exception("3D embedding failed")
    except Exception:
        feature_dict.update({
            'Asphericity': 0,
            'Eccentricity': 0,
            'InertialShapeFactor': 0,
            'RadiusOfGyration': 0,
            'SpherocityIndex': 0,
            'PMI1': 0,
            'PMI2': 0,
            'PMI3': 0,
        })

    # Morgan fingerprint (512 bits, same as your training)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array = np.array(fp)

    for j, bit in enumerate(fp_array):
        feature_dict[f'fp_{j}'] = int(bit)

    return feature_dict

# === ADMET model wrapper ===
class ADMETPredictor:
    """
    Loads pre-saved ADMET models (joblib/.pkl) and their scalers (if any).
    Exposes predict(smiles_list) => DataFrame with columns:
      ['solubility','logp','ames','herg'] and optional probability columns
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        # expected names (change if your filenames differ)
        self.models = {
            'solubility': None,   # regressor
            'logp': None,         # regressor
            'ames': None,         # classifier
            'herg': None          # classifier
        }
        self.scalers = {}  # optional scalers keyed by task name
        self._load_models()

    def _maybe_load(self, pattern_list):
        """return first match joblib.load or None"""
        for pat in pattern_list:
            files = glob.glob(os.path.join(self.model_dir, pat))
            if files:
                return joblib.load(files[0])
        return None

    def _load_models(self):
        # load regressors/regression scalers
        self.models['solubility'] = self._maybe_load(["*solubility*model*.pkl", "*solubility*model*.joblib", "*sol*model*.pkl"])
        self.scalers['solubility'] = self._maybe_load(["*solubility*scaler*.pkl", "*solubility*scaler*.joblib", "*sol*scaler*.pkl"])

        self.models['logp'] = self._maybe_load(["*logp*model*.pkl", "*logp*model*.joblib"])
        self.scalers['logp'] = self._maybe_load(["*logp*scaler*.pkl", "*logp*scaler*.joblib"])

        # classifiers
        self.models['ames'] = self._maybe_load(["*ames*model*.pkl", "*ames*model*.joblib"])
        # sometimes you may have probability-calibrated classifier; we rely on predict_proba if available

        self.models['herg'] = self._maybe_load(["*herg*model*.pkl", "*herg*model*.joblib"])

        # fallback: print what we found
        print("ADMET models loaded:")
        for k,v in self.models.items():
            print(f" - {k}: {'found' if v is not None else 'NOT found'}; scaler: {'found' if self.scalers.get(k) is not None else 'None'}")

    def predict(self, smiles_list: List[str], featurize_func) -> pd.DataFrame:
        rows = []
        feature_names = None
        for smi in smiles_list:
            feats = featurize_func(smi)
            if feats is None:
                rows.append(None)
                continue
            if feature_names is None:
                feature_names = sorted(feats.keys())
            rows.append([feats[n] for n in feature_names])

        # build DataFrame
        if feature_names is None:
            return pd.DataFrame()  # no valid smiles

        X = pd.DataFrame(rows, columns=feature_names)

        out = pd.DataFrame(index=range(len(X)))
        # SOLUBILITY (regression)
        if self.models['solubility'] is not None:
            model = self.models['solubility']
            X_in = X.copy()
            if self.scalers.get('solubility') is not None:
                scaler = self.scalers['solubility']
                try:
                    X_in = scaler.transform(X_in)
                except Exception:
                    # If scaler expects subset of features, try selecting numeric columns
                    X_in = scaler.transform(X_in.select_dtypes(include=[np.number]))
            preds = model.predict(X_in)
            out['solubility'] = preds

        # LOGP (regression)
        if self.models['logp'] is not None:
            model = self.models['logp']
            X_in = X.copy()
            if self.scalers.get('logp') is not None:
                scaler = self.scalers['logp']
                try:
                    X_in = scaler.transform(X_in)
                except Exception:
                    X_in = scaler.transform(X_in.select_dtypes(include=[np.number]))
            preds = model.predict(X_in)
            out['logp'] = preds
        else:
            # fallback: structural RDKit LogP from featurizer (if present)
            if 'LogP' in X.columns:
                out['logp'] = X['LogP'].values

        # AMES (classification)
        if self.models['ames'] is not None:
            model = self.models['ames']
            try:
                # try using featurizer columns the model expects: we attempt to transform with available scaler if exists
                if hasattr(model, "predict_proba"):
                    # if model expects scaled input but no scaler provided, feed raw features; else use scaler if you saved one
                    X_in = X.copy()
                    if self.scalers.get('ames') is not None:
                        X_in = self.scalers['ames'].transform(X_in)
                    probs = model.predict_proba(X_in)
                    # take positive class probability (assume class 1 is positive)
                    out['ames_probability'] = probs[:, 1]
                    out['ames'] = (probs[:, 1] >= 0.5).astype(int)
                else:
                    preds = model.predict(X)
                    out['ames'] = preds
            except Exception:
                # fallback: single-class predictions
                try:
                    preds = model.predict(X.select_dtypes(include=[np.number]))
                    out['ames'] = preds
                except Exception:
                    out['ames'] = 0

        # hERG (classification)
        if self.models['herg'] is not None:
            model = self.models['herg']
            try:
                if hasattr(model, "predict_proba"):
                    X_in = X.copy()
                    if self.scalers.get('herg') is not None:
                        X_in = self.scalers['herg'].transform(X_in)
                    probs = model.predict_proba(X_in)
                    out['herg_probability'] = probs[:, 1]
                    out['herg'] = (probs[:, 1] >= 0.5).astype(int)
                else:
                    preds = model.predict(X)
                    out['herg'] = preds
            except Exception:
                try:
                    preds = model.predict(X.select_dtypes(include=[np.number]))
                    out['herg'] = preds
                except Exception:
                    out['herg'] = 0

        return out.reset_index(drop=True)


# === Bioactivity loader (loads many .joblib models) ===
def load_bioactivity_models(models_dir: str) -> Dict[str, Dict]:
    """
    Scans a directory for joblib/.joblib files (your target-specific files)
    and returns a dict {name: {'model': model_object}}
    Assumes filenames encode the target name (e.g., cancer_EGFR_classification_model.joblib)
    """
    bio = {}
    patterns = ["*.joblib", "*.pkl", "*.model", "*.model.pkl", "*.joblib"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(models_dir, pat)))
    files = sorted(set(files))
    for f in files:
        name = os.path.basename(f)
        # try to extract a clean target name
        target = name.replace(".joblib", "").replace(".pkl", "").replace(".model", "")
        target = target.replace("_classification", "").replace("_regression", "")
        # load
        try:
            m = joblib.load(f)
            bio[target] = {'model': m}
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            continue
    print(f"Loaded {len(bio)} bioactivity models.")
    return bio


# === Modified MoleculeOptimizationAgent that uses the above ADMETPredictor & bioactivity dict ===
from rdkit.Chem import BRICS

class MoleculeOptimizationAgent:
    def __init__(self, admet_predictor: ADMETPredictor, bioactivity_models: Optional[Dict]=None, plant_database: Optional[pd.DataFrame]=None, docking_df: Optional[pd.DataFrame] = None):
        self.admet_predictor = admet_predictor
        self.bioactivity_models = bioactivity_models or {}
        self.plant_database = plant_database
        self.docking_df = docking_df

    def generate_analogs(self, smiles: str, n_analogs: int = 50) -> List[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        try:
            frags = list(BRICS.BRICSDecompose(mol))
            frag_mols = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f)]
        except Exception:
            return []
        if len(frag_mols) < 1:
            return []
        analogs = set()
        attempts = 0
        while len(analogs) < n_analogs and attempts < n_analogs * 10:
            attempts += 1
            # randomly select 2-4 fragments
            n_frags = np.random.randint(1, min(4, len(frag_mols)) + 1)
            selected = list(np.random.choice(frag_mols, size=n_frags, replace=True))
            try:
                new = selected[0]
                for frag in selected[1:]:
                    new = Chem.CombineMols(new, frag)
                Chem.SanitizeMol(new)
                s = Chem.MolToSmiles(new, isomericSmiles=True)
                if self._is_druglike(s) and s != smiles:
                    analogs.add(s)
            except Exception:
                continue
        return list(analogs)

    def _is_druglike(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

    def score_admet(self, smiles: str, featurize_func) -> Dict[str, float]:
        df = self.admet_predictor.predict([smiles], featurize_func)
        if df.shape[0] == 0:
            return {}
        row = df.iloc[0]
        scores = {}
        # Normalization heuristics (same as original agent)
        if 'solubility' in row:
            sol = float(row['solubility'])
            scores['solubility'] = min(1.0, max(0.0, (sol + 5) / 6))
        if 'logp' in row:
            logp = float(row['logp'])
            s = 1.0 - abs(logp - 2.5) / 5.0
            scores['logp'] = max(0.0, s)
        if 'herg_probability' in row:
            scores['herg'] = 1.0 - float(row['herg_probability'])
        elif 'herg' in row:
            scores['herg'] = 1.0 - float(row['herg'])
        if 'ames_probability' in row:
            scores['ames'] = 1.0 - float(row['ames_probability'])
        elif 'ames' in row:
            scores['ames'] = 1.0 - float(row['ames'])
        return scores

    def score_bioactivity(self, smiles: str, featurize_func) -> Dict[str, float]:
        if not self.bioactivity_models:
            return {}
        feats = featurize_func(smiles)
        if feats is None:
            return {}
        X = pd.DataFrame([feats])
        scores = {}
        for target, md in self.bioactivity_models.items():
            model = md.get('model')
            if model is None:
                continue
            try:
                # try predict_proba
                if hasattr(model, "predict_proba"):
                    # model may expect subset; attempt direct predict_proba with X
                    p = model.predict_proba(X)[0][1]
                    scores[target] = float(p)
                else:
                    # regression
                    pred = model.predict(X)[0]
                    scores[target] = float(np.clip(pred / 10.0, 0.0, 1.0))
            except Exception:
                # try numeric-only fallback
                try:
                    p = model.predict(X.select_dtypes(include=[np.number]))
                    if hasattr(model, "predict_proba"):
                        scores[target] = float(p[0][1])
                    else:
                        scores[target] = float(np.clip(p[0]/10.0, 0.0, 1.0))
                except Exception:
                    continue
        return scores

    def multi_objective_score(self, smiles: str, featurize_func, weights: Dict[str, float] = None) -> Tuple[float, Dict]:
        if weights is None:
            weights = {'admet': 0.4, 'bioactivity': 0.4, 'druglikeness': 0.2}
        admet_scores = self.score_admet(smiles, featurize_func)
        bio_scores = self.score_bioactivity(smiles, featurize_func)
        admet_avg = np.mean(list(admet_scores.values())) if admet_scores else 0.0
        bio_avg = np.mean(list(bio_scores.values())) if bio_scores else 0.5
        druglike_score = 1.0 if self._is_druglike(smiles) else 0.3
        total = weights['admet']*admet_avg + weights['bioactivity']*bio_avg + weights['druglikeness']*druglike_score
        detailed = {'total_score': total, 'admet_avg': admet_avg, 'bioactivity_avg': bio_avg, 'druglikeness': druglike_score}
        detailed.update(admet_scores)
        detailed.update(bio_scores)
        return total, detailed

    def optimize_molecule(self, parent_smiles: str, featurize_func, n_analogs: int = 100, top_n: int = 5) -> pd.DataFrame:
        print(f"Generating up to {n_analogs} analogs for parent molecule...")
        analogs = self.generate_analogs(parent_smiles, n_analogs)
        if len(analogs) == 0:
            analogs = [parent_smiles]
        results = []
        all_mols = [parent_smiles] + analogs
        for i, s in enumerate(all_mols):
            try:
                total, details = self.multi_objective_score(s, featurize_func)
                details['smiles'] = s
                details['is_parent'] = (s == parent_smiles)
                results.append(details)
            except Exception:
                continue
        df = pd.DataFrame(results).sort_values('total_score', ascending=False).reset_index(drop=True)
        if self.plant_database is not None and 'smiles' in self.plant_database.columns:
            df = df.merge(self.plant_database[['smiles','plant_name','compound_name']], on='smiles', how='left')
            df['plant_origin'] = df['plant_name'].fillna('Synthetic analog')
            df['compound_name'] = df['compound_name'].fillna('Novel compound')
        return df.head(top_n)

# === Example usage ===
if __name__ == "__main__":
    # point this to your models folder
    MODELS_DIR = "models"   # change if different

    # 1) ADMET predictor (your .pkl models + optional scalers)
    admet = ADMETPredictor(MODELS_DIR)

    # 2) Load bioactivity models (.joblib or .pkl)
    bio_models = load_bioactivity_models(MODELS_DIR)

    # 3) (optional) load plant DB (if you have one)
    # plant_db = pd.read_csv("plant_db.csv")  # must contain columns: smiles, plant_name, compound_name
    plant_db = None

    # 4) Initialize optimizer
    optimizer = MoleculeOptimizationAgent(admet, bio_models, plant_db)

    # 5) Optimize a molecule
    parent = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"  # example
    top_df = optimizer.optimize_molecule(parent, featurize, n_analogs=100, top_n=5)
    print(top_df)

    # 6) Save results
    top_df.to_csv("optimized_molecules.csv", index=False)

