import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

st.set_page_config(page_title="SMARTCyp Pro v2", layout="wide")

# -------------------------------
# 0. SAFE UTILITIES
# -------------------------------
def safe_shortest_path_length(mol, idx1, idx2):
    try:
        path = Chem.GetShortestPath(mol, idx1, idx2)
        return len(path) if path else 999
    except:
        return 999


# -------------------------------
# 1. ATOM TYPING
# -------------------------------
def get_atom_type(atom):
    symbol = atom.GetSymbol()

    if atom.GetIsAromatic() and symbol == "C":
        return ("Aromatic_C", 62.0)

    if symbol == "C":
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            h = atom.GetTotalNumHs()
            if h == 3:
                return ("Primary_C", 55.0)
            elif h == 2:
                return ("Secondary_C", 50.0)
            elif h == 1:
                return ("Tertiary_C", 48.0)

    if symbol == "N":
        return ("Amine_N", 45.0)

    if symbol == "O":
        return ("Oxygen", 52.0)

    return ("Other", 80.0)


# -------------------------------
# 2. ACCESSIBILITY
# -------------------------------
def accessibility_score(atom):
    degree = atom.GetDegree()
    in_ring = atom.IsInRing()

    penalty = 0

    if degree >= 3:
        penalty += 5

    if in_ring:
        penalty += 3

    return penalty


# -------------------------------
# 3. ANALYSIS ENGINE
# -------------------------------
def analyze_isoform(mol, isoform_type):
    results = []

    # Safe charge calculation
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    anchors_2d6 = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[NX3;!$(NC=O)]")
    )

    anchors_2c9 = mol.GetSubstructMatches(
        Chem.MolFromSmarts("C(=O)[O]")
    )

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        atom_type, base_energy = get_atom_type(atom)
        acc = accessibility_score(atom)

        score = base_energy + acc

        # Charge contribution
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if charge == charge:  # avoid NaN
                score += abs(charge) * 5
        except:
            pass

        # Isoform corrections
        if isoform_type == "CYP2D6" and anchors_2d6:
            distances = [
                safe_shortest_path_length(mol, idx, a[0])
                for a in anchors_2d6
            ]
            score += min(distances) * 1.5

        elif isoform_type == "CYP2C9" and anchors_2c9:
            distances = [
                safe_shortest_path_length(mol, idx, a[0])
                for a in anchors_2c9
            ]
            score += min(distances) * 1.2

        results.append({
            "Atom": idx + 1,
            "Type": atom_type,
            "Score": round(score, 2)
        })

    df = pd.DataFrame(results)

    # Safe normalization
    min_s = df["Score"].min()
    max_s = df["Score"].max()

    if max_s - min_s < 1e-6:
        df["NormScore"] = 0.0
    else:
        df["NormScore"] = (df["Score"] - min_s) / (max_s - min_s)

    return df.sort_values("Score")


# -------------------------------
# 4. UI
# -------------------------------
st.title("🧪 SMARTCyp Pro v2 (Python Edition)")

with st.sidebar:
    st.header("Settings")
    selected_isoform = st.selectbox(
        "Select Isoform:",
        ["CYP3A4", "CYP2D6", "CYP2C9"]
    )
    st.info(f"Analyzing for **{selected_isoform}**")


# -------------------------------
# 5. INPUT (Enhanced)
# -------------------------------
st.subheader("Molecule Input")

input_mode = st.radio(
    "Choose input method:",
    ["Draw Molecule", "Enter SMILES"]
)

smiles = None

if input_mode == "Draw Molecule":
    molecule_smiles = st_ketcher(key="ketcher_editor")

    if molecule_smiles:
        smiles = molecule_smiles
    else:
        st.caption("Draw a molecule to begin")

elif input_mode == "Enter SMILES":
    smiles_input = st.text_input(
        "Enter SMILES:",
        value="CNC1=CC=C(C=C1)C2=CC=CC=C2"
    )

    if smiles_input:
        smiles = smiles_input.strip()


# -------------------------------
# 6. ANALYSIS
# -------------------------------
if smiles:
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # Keep largest fragment only
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda m: m.GetNumAtoms())

        tab1, tab2, tab3 = st.tabs(
            ["📊 Single Isoform", "🏁 Comparison", "🧊 3D View"]
        )

        # TAB 1
        with tab1:
            df = analyze_isoform(mol, selected_isoform)

            if not df.empty:
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Top Metabolic Sites")

                    top_atoms = [
                        int(x - 1) for x in df.head(3)["Atom"]
                    ]

                    img = Draw.MolToImage(
                        mol,
                        size=(400, 400),
                        highlightAtoms=top_atoms
                    )
                    st.image(img)

                with c2:
                    st.subheader("Scores")
                    st.dataframe(df, use_container_width=True)

        # TAB 2
        with tab2:
            cols = st.columns(3)

            for i, iso in enumerate(["CYP3A4", "CYP2D6", "CYP2C9"]):
                with cols[i]:
                    st.markdown(f"**{iso}**")

                    df_iso = analyze_isoform(mol, iso)

                    if not df_iso.empty:
                        top_atoms = [
                            int(x - 1)
                            for x in df_iso.head(3)["Atom"]
                        ]

                        img = Draw.MolToImage(
                            mol,
                            size=(300, 300),
                            highlightAtoms=top_atoms
                        )
                        st.image(img)
                        st.dataframe(df_iso.head(5))

        # TAB 3
        with tab3:
            st.subheader("3D Structure")

            m3d = Chem.AddHs(mol)

            if AllChem.EmbedMolecule(m3d, AllChem.ETKDG()) == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(m3d)
                except:
                    pass

                mblock = Chem.MolToMolBlock(m3d)

                view = py3Dmol.view(width=800, height=500)
                view.addModel(mblock, 'mol')
                view.setStyle({'stick': {}})
                view.zoomTo()

                showmol(view, height=500, width=800)
            else:
                st.warning("3D generation failed for this molecule")

    else:
        st.error("Invalid SMILES")


# -------------------------------
# FOOTER
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
SMARTCyp Pro v2  
- Atom-type scoring  
- Accessibility correction  
- Charge contribution  
- Isoform-aware penalties  
- Stable & cloud-ready  
""")