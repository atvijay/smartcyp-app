import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdChemReactions
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

st.set_page_config(page_title="SMARTCyp Pro v3.1", layout="wide")

# -------------------------------
# UTILITIES
# -------------------------------
def safe_shortest_path_length(mol, idx1, idx2):
    try:
        path = Chem.GetShortestPath(mol, idx1, idx2)
        return len(path) if path else 999
    except:
        return 999


# -------------------------------
# ATOM TYPING
# -------------------------------
def get_atom_type(atom):
    if atom.GetIsAromatic() and atom.GetSymbol() == "C":
        return ("Aromatic_C", 62.0)

    if atom.GetSymbol() == "C":
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            h = atom.GetTotalNumHs()
            if h == 3: return ("Primary_C", 55.0)
            elif h == 2: return ("Secondary_C", 50.0)
            elif h == 1: return ("Tertiary_C", 48.0)

    if atom.GetSymbol() == "N": return ("Amine_N", 45.0)
    if atom.GetSymbol() == "O": return ("Oxygen", 52.0)

    return ("Other", 80.0)


def accessibility_score(atom):
    score = 0
    if atom.GetDegree() >= 3: score += 5
    if atom.IsInRing(): score += 3
    return score


# -------------------------------
# SMARTCYP SCORING
# -------------------------------
def analyze_isoform(mol, isoform):
    results = []

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    anchors_2d6 = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;!$(NC=O)]"))
    anchors_2c9 = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[O]"))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        atom_type, base = get_atom_type(atom)
        score = base + accessibility_score(atom)

        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if charge == charge:
                score += abs(charge) * 5
        except:
            pass

        if isoform == "CYP2D6" and anchors_2d6:
            score += min([safe_shortest_path_length(mol, idx, a[0]) for a in anchors_2d6]) * 1.5

        if isoform == "CYP2C9" and anchors_2c9:
            score += min([safe_shortest_path_length(mol, idx, a[0]) for a in anchors_2c9]) * 1.2

        results.append({
            "Atom": idx+1,
            "Type": atom_type,
            "Score": round(score, 2)
        })

    df = pd.DataFrame(results)

    df["NormScore"] = (
        (df["Score"] - df["Score"].min()) /
        (df["Score"].max() - df["Score"].min() + 1e-6)
    )

    return df.sort_values("Score")


# -------------------------------
# METABOLITE GENERATION (v3)
# -------------------------------
def generate_metabolites_v3(mol, df):
    metabolites = []

    for _, row in df.head(5).iterrows():
        atom_idx = int(row["Atom"] - 1)
        atom = mol.GetAtomWithIdx(atom_idx)
        base_score = row["NormScore"]

        # Hydroxylation
        if atom.GetSymbol() == "C":
            m = Chem.RWMol(mol)
            o_idx = m.AddAtom(Chem.Atom("O"))
            m.AddBond(atom_idx, o_idx, Chem.BondType.SINGLE)
            try:
                metabolites.append({
                    "Reaction": "Hydroxylation",
                    "Site": atom_idx+1,
                    "Score": round(base_score, 3),
                    "SMILES": Chem.MolToSmiles(m)
                })
            except:
                pass

        # N-dealkylation
        if atom.GetSymbol() == "N":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "C":
                    m = Chem.RWMol(mol)
                    m.RemoveBond(atom_idx, nbr.GetIdx())
                    try:
                        metabolites.append({
                            "Reaction": "N-dealkylation",
                            "Site": atom_idx+1,
                            "Score": round(base_score*0.9, 3),
                            "SMILES": Chem.MolToSmiles(m)
                        })
                    except:
                        pass
                    break

        # Epoxidation
        if atom.GetHybridization() == Chem.HybridizationType.SP2:
            rxn = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[C:2]>>[C:1]1[C:2]O1"
            )
            try:
                products = rxn.RunReactants((mol,))
                for pset in products[:1]:
                    for p in pset:
                        Chem.SanitizeMol(p)
                        metabolites.append({
                            "Reaction": "Epoxidation",
                            "Site": atom_idx+1,
                            "Score": round(base_score*0.8, 3),
                            "SMILES": Chem.MolToSmiles(p)
                        })
            except:
                pass

    return pd.DataFrame(metabolites).drop_duplicates()


# -------------------------------
# AGENTIC OPTIMIZATION
# -------------------------------
def suggest_modifications(mol, df):
    suggestions = []

    for _, row in df.head(3).iterrows():
        atom_idx = int(row["Atom"] - 1)
        atom = mol.GetAtomWithIdx(atom_idx)

        # Benzylic methylation
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    m = Chem.RWMol(mol)
                    c_idx = m.AddAtom(Chem.Atom("C"))
                    m.AddBond(atom_idx, c_idx, Chem.BondType.SINGLE)

                    try:
                        suggestions.append({
                            "Strategy": "Benzylic methylation",
                            "Site": atom_idx+1,
                            "SMILES": Chem.MolToSmiles(m)
                        })
                    except:
                        pass

        # Aromatic fluorination
        if atom.GetIsAromatic():
            m = Chem.RWMol(mol)
            f_idx = m.AddAtom(Chem.Atom("F"))
            m.AddBond(atom_idx, f_idx, Chem.BondType.SINGLE)

            try:
                suggestions.append({
                    "Strategy": "Aromatic fluorination",
                    "Site": atom_idx+1,
                    "SMILES": Chem.MolToSmiles(m)
                })
            except:
                pass

    return pd.DataFrame(suggestions).drop_duplicates()


# -------------------------------
# UI
# -------------------------------
st.title("🧪 SMARTCyp Pro v3.1")

iso = st.sidebar.selectbox(
    "Isoform",
    ["CYP3A4","CYP2D6","CYP2C9"]
)

mode = st.radio("Input:", ["Draw","SMILES"])
smiles = None

if mode == "Draw":
    s = st_ketcher()
    if s:
        smiles = s
else:
    s = st.text_input(
        "Enter SMILES",
        "CNC1=CC=C(C=C1)C2=CC=CC=C2"
    )
    if s:
        smiles = s.strip()

# -------------------------------
# MAIN APP
# -------------------------------
if smiles:
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        mol = max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda m: m.GetNumAtoms()
        )

        df = analyze_isoform(mol, iso)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Analysis","3D","Metabolites","Optimization"]
        )

        # Analysis
        with tab1:
            st.dataframe(df)

            img = Draw.MolToImage(
                mol,
                highlightAtoms=[int(x-1) for x in df.head(3)["Atom"]],
                size=(400,400)
            )
            st.image(img)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "results.csv"
            )

        # 3D
        with tab2:
            m3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(m3d) == 0:
                view = py3Dmol.view(width=600, height=400)
                view.addModel(Chem.MolToMolBlock(m3d), "mol")
                view.setStyle({"stick": {}})
                view.zoomTo()
                showmol(view)

        # Metabolites
        with tab3:
            met_df = generate_metabolites_v3(mol, df)

            if not met_df.empty:
                st.dataframe(met_df)

                for _, row in met_df.iterrows():
                    st.write(f"{row['Reaction']} (Site {row['Site']})")

                    m = Chem.MolFromSmiles(row["SMILES"])
                    if m:
                        st.image(Draw.MolToImage(m, size=(250,250)))

                st.download_button(
                    "Download Metabolites",
                    met_df.to_csv(index=False),
                    "metabolites.csv"
                )
            else:
                st.info("No metabolites generated")

        # Optimization
        with tab4:
            st.subheader("🧠 Metabolism-Guided Optimization")

            opt_df = suggest_modifications(mol, df)

            if not opt_df.empty:
                st.dataframe(opt_df)

                for _, row in opt_df.iterrows():
                    st.write(f"{row['Strategy']} (Site {row['Site']})")

                    m = Chem.MolFromSmiles(row["SMILES"])
                    if m:
                        st.image(Draw.MolToImage(m, size=(250,250)))

                        # Evaluate improvement
                        new_df = analyze_isoform(m, iso)
                        if new_df["Score"].min() < df["Score"].min():
                            st.success("Improved metabolic stability ✅")
                        else:
                            st.warning("No improvement ⚠️")

                st.download_button(
                    "Download Optimized Molecules",
                    opt_df.to_csv(index=False),
                    "optimized.csv"
                )
            else:
                st.info("No optimization suggestions")