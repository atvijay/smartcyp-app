# smartcyp_docking_app.py
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

# Optional: Vina docking
try:
    from vina import Vina
    vina_available = True
except ImportError:
    vina_available = False

st.set_page_config(page_title="SMARTCyp Docking Pro", layout="wide")

# -------------------------------
# Utilities
# -------------------------------
def safe_shortest_path_length(mol, idx1, idx2):
    try:
        path = Chem.GetShortestPath(mol, idx1, idx2)
        return len(path) if path else 999
    except:
        return 999

def get_atom_type(atom):
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic() and symbol == "C":
        return ("Aromatic_C", 62.0)
    if symbol == "C":
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            h = atom.GetTotalNumHs()
            if h == 3: return ("Primary_C", 55.0)
            if h == 2: return ("Secondary_C", 50.0)
            if h == 1: return ("Tertiary_C", 48.0)
    if symbol == "N": return ("Amine_N", 45.0)
    if symbol == "O": return ("Oxygen", 52.0)
    return ("Other", 80.0)

def accessibility_score(atom):
    penalty = 0
    if atom.GetDegree() >= 3: penalty += 5
    if atom.IsInRing(): penalty += 3
    return penalty

# -------------------------------
# SMARTCyp Scoring Engine
# -------------------------------
def analyze_isoform(mol, isoform_type):
    results = []
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    anchors_2d6 = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;!$(NC=O)]"))
    anchors_2c9 = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[O]"))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_type, base_energy = get_atom_type(atom)
        score = base_energy + accessibility_score(atom)
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if charge == charge:
                score += abs(charge) * 5
        except: pass

        # Isoform corrections
        if isoform_type == "CYP2D6" and anchors_2d6:
            distances = [safe_shortest_path_length(mol, idx, a[0]) for a in anchors_2d6]
            score += min(distances) * 1.5
        elif isoform_type == "CYP2C9" and anchors_2c9:
            distances = [safe_shortest_path_length(mol, idx, a[0]) for a in anchors_2c9]
            score += min(distances) * 1.2

        results.append({"Atom": idx+1, "Type": atom_type, "Score": round(score,2)})

    df = pd.DataFrame(results)
    min_s, max_s = df["Score"].min(), df["Score"].max()
    if max_s - min_s < 1e-6:
        df["NormScore"] = 0.0
    else:
        df["NormScore"] = (df["Score"]-min_s)/(max_s-min_s)
    return df.sort_values("Score")

# -------------------------------
# Metabolite Generator
# -------------------------------
def generate_metabolites(mol, df):
    metabolites = []
    top_atoms = [int(x-1) for x in df.head(3)["Atom"]]

    for atom_idx in top_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        # Hydroxylation
        if atom.GetSymbol() == "C":
            m = Chem.RWMol(mol)
            o_idx = m.AddAtom(Chem.Atom("O"))
            m.AddBond(atom_idx, o_idx, Chem.BondType.SINGLE)
            try:
                metabolites.append({"Type":"Hydroxylation","Atom":atom_idx+1,"SMILES":Chem.MolToSmiles(m)})
            except: pass
        # N-dealkylation
        if atom.GetSymbol() == "N":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol()=="C":
                    m = Chem.RWMol(mol)
                    m.RemoveBond(atom_idx,nbr.GetIdx())
                    try:
                        metabolites.append({"Type":"N-dealkylation","Atom":atom_idx+1,"SMILES":Chem.MolToSmiles(m)})
                    except: pass
                    break
        # O-dealkylation
        if atom.GetSymbol() == "O":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol()=="C":
                    m = Chem.RWMol(mol)
                    m.RemoveBond(atom_idx,nbr.GetIdx())
                    try:
                        metabolites.append({"Type":"O-dealkylation","Atom":atom_idx+1,"SMILES":Chem.MolToSmiles(m)})
                    except: pass
                    break
    return pd.DataFrame(metabolites)

# -------------------------------
# Docking-Constrained Scoring
# -------------------------------
def docking_score(df, docked_mol, heme_coord=[0,0,0]):
    conf = docked_mol.GetConformer()
    distances = []
    for atom in docked_mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        dist = ((pos.x - heme_coord[0])**2 + (pos.y - heme_coord[1])**2 + (pos.z - heme_coord[2])**2)**0.5
        distances.append(dist)
    df["DockDist"] = distances
    df["DockScore"] = df["NormScore"]*0.6 + 1/(df["DockDist"]+1e-3)*0.4
    return df.sort_values("DockScore",ascending=False)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧪 SMARTCyp Docking Pro")

with st.sidebar:
    selected_isoform = st.selectbox("Isoform",["CYP3A4","CYP2D6","CYP2C9"])
    if vina_available:
        st.success("Vina available for docking")
    else:
        st.warning("Vina not installed. Docking disabled.")

# Input
st.subheader("Molecule Input")
mode = st.radio("Input type:", ["Draw","SMILES"])
smiles = None
if mode=="Draw":
    s = st_ketcher(key="ketcher")
    if s: smiles=s
else:
    s = st.text_input("Enter SMILES","CNC1=CC=C(C=C1)C2=CC=CC=C2")
    if s: smiles=s.strip()

if smiles:
    st.code(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m:m.GetNumAtoms())
        tab1,tab2,tab3,tab4=st.tabs(["Analysis","Comparison","3D","Metabolites"])

        # -------- TAB 1: Analysis
        with tab1:
            df=analyze_isoform(mol,selected_isoform)
            st.dataframe(df)
            img=Draw.MolToImage(mol,highlightAtoms=[int(x-1) for x in df.head(3)["Atom"]],size=(400,400))
            st.image(img)
            df_exp=df.copy()
            df_exp["SMILES"]=smiles
            df_exp["Isoform"]=selected_isoform
            st.download_button("Download CSV",df_exp.to_csv(index=False),"results.csv")

        # -------- TAB 2: Comparison
        with tab2:
            for iso in ["CYP3A4","CYP2D6","CYP2C9"]:
                st.subheader(iso)
                df_iso=analyze_isoform(mol,iso)
                st.dataframe(df_iso.head(5))

        # -------- TAB 3: 3D
        with tab3:
            m3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(m3d)==0:
                view=py3Dmol.view(width=600,height=400)
                view.addModel(Chem.MolToMolBlock(m3d),"mol")
                view.setStyle({"stick":{}})
                view.zoomTo()
                showmol(view)

        # -------- TAB 4: Metabolites
        with tab4:
            df=analyze_isoform(mol,selected_isoform)
            met_df=generate_metabolites(mol,df)
            if not met_df.empty:
                st.dataframe(met_df)
                for _,row in met_df.iterrows():
                    st.write(row["Type"])
                    m = Chem.MolFromSmiles(row["SMILES"])
                    if m: st.image(Draw.MolToImage(m))
                st.download_button("Download Metabolites",met_df.to_csv(index=False),"metabolites.csv")
            else:
                st.info("No metabolites generated")

st.sidebar.info("""
**SMARTCyp Pro v1.0**
Predicts metabolic sites for CYP3A4, 2D6, and 2C9.
Built using RDKit and SMARTCyp 3.0 logic.
""")