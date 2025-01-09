import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# RDKit for molecule handling
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# For generating images in Streamlit
from PIL import Image

# Suppress warnings in RDKit
import warnings
warnings.filterwarnings('ignore')

# Set Seaborn style
sns.set_style('whitegrid')

# Additional imports for GNN
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

# Function to load the VAE model
@st.cache_resource
def load_vae_model(device):
    # Load the vocabulary
    with open('vae_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    
    # Initialize the model with the same parameters
    hidden_dim = 256  # Ensure this matches your trained model
    latent_dim = 64   # Ensure this matches your trained model
    
    # Define the VAE class (same as in your training script)
    class VAE(nn.Module):
        def __init__(self, vocab_size: int, hidden_dim: int, latent_dim: int):
            super(VAE, self).__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            self.encoder = nn.GRU(vocab_size, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.GRU(vocab_size + latent_dim, hidden_dim, batch_first=True)
            self.fc_output = nn.Linear(hidden_dim, vocab_size)
        
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            _, h = self.encoder(x)
            h = h.squeeze(0)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor, max_length: int) -> torch.Tensor:
            batch_size = z.size(0)
            h = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
            x = torch.zeros(batch_size, 1, self.vocab_size).to(z.device)
            x[:, 0, vocab['<']] = 1  # Start token
            outputs = []

            for _ in range(max_length):
                z_input = z.unsqueeze(1)
                decoder_input = torch.cat([x, z_input], dim=2)
                output, h = self.decoder(decoder_input, h)
                output = self.fc_output(output)
                outputs.append(output)
                x = torch.softmax(output, dim=-1)

            return torch.cat(outputs, dim=1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, x.size(1)), mu, logvar

    model = VAE(vocab_size, hidden_dim, latent_dim)
    model.load_state_dict(torch.load('vae_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, vocab

# Function to generate molecules using VAE
def generate_smiles_vae(model, vocab, num_samples=10, max_length=100):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    generated_smiles = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, model.latent_dim).to(device)
            x = torch.zeros(1, 1, model.vocab_size).to(device)
            x[0, 0, vocab['<']] = 1
            h = torch.zeros(1, 1, model.hidden_dim).to(device)

            smiles = ''
            for _ in range(max_length):
                z_input = z.unsqueeze(1)
                decoder_input = torch.cat([x, z_input], dim=2)
                output, h = model.decoder(decoder_input, h)
                output = model.fc_output(output)

                probs = torch.softmax(output.squeeze(0), dim=-1)
                next_char = torch.multinomial(probs, 1).item()

                if next_char == vocab['>']:
                    break

                smiles += inv_vocab.get(next_char, '')
                x = torch.zeros(1, 1, model.vocab_size).to(device)
                x[0, 0, next_char] = 1

            generated_smiles.append(smiles)

    return generated_smiles

# Function to post-process and validate SMILES strings
def enhanced_post_process_smiles(smiles: str) -> str:
    smiles = smiles.replace('<', '').replace('>', '')
    allowed_chars = set('CNOPSFIBrClcnops()[]=@+-#0123456789')
    smiles = ''.join(c for c in smiles if c in allowed_chars)

    # Balance parentheses
    open_count = smiles.count('(')
    close_count = smiles.count(')')
    if open_count > close_count:
        smiles += ')' * (open_count - close_count)
    elif close_count > open_count:
        smiles = '(' * (close_count - open_count) + smiles

    # Replace invalid double bonds
    smiles = smiles.replace('==', '=')

    # Attempt to close unclosed rings
    for i in range(1, 10):
        if smiles.count(str(i)) % 2 != 0:
            smiles += str(i)

    return smiles

def validate_and_correct_smiles(smiles: str) -> Tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            return True, Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            pass
    return False, smiles

# Function to analyze molecules
def analyze_molecules(smiles_list: List[str], training_smiles_set: set) -> Dict:
    results = {
        'total': len(smiles_list),
        'valid': 0,
        'invalid': 0,
        'unique': 0,
        'corrected': 0,
        'novel': 0,
        'valid_properties': [],
        'novel_properties': [],
        'invalid_smiles': []
    }

    unique_smiles = set()
    novel_smiles = set()

    for smiles in smiles_list:
        processed_smiles = enhanced_post_process_smiles(smiles)
        is_valid, corrected_smiles = validate_and_correct_smiles(processed_smiles)

        if is_valid:
            results['valid'] += 1
            unique_smiles.add(corrected_smiles)
            if corrected_smiles != processed_smiles:
                results['corrected'] += 1

            mol = Chem.MolFromSmiles(corrected_smiles)
            if mol:
                props = {
                    'smiles': corrected_smiles,
                    'MolWt': Descriptors.ExactMolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol)
                }

                if corrected_smiles not in training_smiles_set:
                    novel_smiles.add(corrected_smiles)
                    results['novel'] += 1
                    results['novel_properties'].append(props)
                else:
                    results['valid_properties'].append(props)
        else:
            results['invalid'] += 1
            results['invalid_smiles'].append(smiles)

    results['unique'] = len(unique_smiles)
    return results

# Function to visualize molecules
def visualize_molecules(smiles_list: List[str], n: int = 5) -> Optional[Image.Image]:
    valid_mols = []
    for smiles in smiles_list:
        smiles = smiles.strip().strip('<>').strip()
        if not smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                if len(valid_mols) == n:
                    break
        except Exception:
            continue

    if not valid_mols:
        return None

    try:
        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=min(3, len(valid_mols)),
            subImgSize=(200, 200),
            legends=[f"Mol {i+1}" for i in range(len(valid_mols))]
        )
        return img
    except Exception:
        return None

# GCN and GIN model definitions
class GCN(torch.nn.Module):
    """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

    def __init__(self, dim_h):
        """init method for GCN

        Args:
            dim_h (int): the dimension of hidden layers
        """
        super().__init__()
        self.conv1 = GCNConv(11, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = torch.nn.Linear(dim_h, 1)

    def forward(self, data):
        e = data.edge_index
        x = data.x

        x = self.conv1(x, e)
        x = x.relu()
        x = self.conv2(x, e)
        x = x.relu()
        x = self.conv3(x, e)
        x = global_mean_pool(x, data.batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GIN(torch.nn.Module):
    """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""

    def __init__(self, dim_h):
        """Initializing GIN class

        Args:
            dim_h (int): the dimension of hidden layers
        """
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(11, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU())
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU())
        self.conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU())
        self.conv3 = GINConv(nn3)
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Node embeddings
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        h = global_add_pool(h, batch)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

# Function to load GNN models
@st.cache_resource
def load_gnn_models(device):
    # Load GCN model
    gcn_model = GCN(dim_h=128)
    gcn_model.load_state_dict(torch.load("GCN_model.pth", map_location=device))
    gcn_model.to(device)
    gcn_model.eval()

    # Load GIN model
    gin_model = GIN(dim_h=64)
    gin_model.load_state_dict(torch.load("GIN_model.pth", map_location=device))
    gin_model.to(device)
    gin_model.eval()

    return gcn_model, gin_model

# Function to load normalization parameters
@st.cache_resource
def load_data_norm(device):
    data_norm = torch.load('data_norm.pth', map_location=device)
    data_mean = data_norm['mean']
    data_std = data_norm['std']
    return data_mean, data_std

# Function to convert SMILES to Data object
def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = mol.GetAtoms()
    num_atoms = len(atoms)

    atom_type_list = ['H', 'C', 'N', 'O', 'F']
    hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]

    x = []
    for atom in atoms:
        atom_type = atom.GetSymbol()
        atom_type_feature = [int(atom_type == s) for s in atom_type_list]  # 5 features

        # Atom degree (scalar between 0 and 4)
        degree = atom.GetDegree()
        degree_feature = [degree / 4]  # Normalize degree to [0,1]  # 1 feature

        # Formal charge
        formal_charge = atom.GetFormalCharge()
        formal_charge_feature = [formal_charge / 4]  # Assume max formal charge is 4  # 1 feature

        # Aromaticity
        is_aromatic = atom.GetIsAromatic()
        aromatic_feature = [int(is_aromatic)]  # 1 feature

        # Hybridization
        hybridization = atom.GetHybridization()
        hybridization_feature = [int(hybridization == hyb) for hyb in hybridization_list]  # 3 features

        # Total features: 5 + 1 +1 +1 +3 = 11
        atom_feature = atom_type_feature + degree_feature + formal_charge_feature + aromatic_feature + hybridization_feature
        x.append(atom_feature)

    x = torch.tensor(x, dtype=torch.float)

    # Build edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Since it's undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Build batch tensor (since batch size is 1)
    batch = torch.zeros(num_atoms, dtype=torch.long)

    # Build Data object
    data = Data(x=x, edge_index=edge_index, batch=batch)

    return data

# Streamlit app
def main():
    st.set_page_config(
        page_title="🧪 Molecule Generator and Property Predictor",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Main Title and Description
    st.title("🧪 Molecular Generation and Analysis using VAE and GNN")
    st.markdown("""
    SMILES (Simplified Molecular Input Line Entry System) is a widely-used notation that encodes chemical structures into short, linear strings of characters. 
    This representation allows for the easy storage, transmission, and manipulation of molecular information in computational applications.
   
    This application allows you to generate novel molecular SMILES structures using a Variational Autoencoder (VAE) model trained on the QM9 dataset.
    You can also predict molecular properties using Graph Neural Network (GNN) models (GCN and GIN).
    """)

    # Initialize session state variables
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'generated_smiles' not in st.session_state:
        st.session_state.generated_smiles = []
    if 'vae_generated' not in st.session_state:
        st.session_state.vae_generated = False

    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    st.sidebar.markdown("Adjust the settings below to generate molecules or predict properties.")

    # Load training data and canonicalize SMILES
    @st.cache_data
    def load_training_data():
        df = pd.read_csv("qm9.csv")
        smiles_list_raw = df['smiles'].tolist()
        # Canonicalize SMILES for accurate comparison
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True) for s in smiles_list_raw]
        return set(smiles_list)

    training_smiles_set = load_training_data()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    st.sidebar.title("📌 Model Selection")
    model_option = st.sidebar.selectbox("Choose a functionality", ("Generate Molecules (VAE)", "Predict Property (GNN)"))

    if model_option == "Generate Molecules (VAE)":
        # Number of samples
        num_samples = st.sidebar.slider("Number of Molecules to Generate", min_value=5, max_value=500, value=50, step=5)

        # Random seed
        seed = st.sidebar.number_input("Random Seed", value=42, step=1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if st.sidebar.button("🚀 Generate Molecules"):
            with st.spinner("Generating molecules..."):
                # Load VAE model
                model, vocab = load_vae_model(device)
                generated_smiles = generate_smiles_vae(model, vocab, num_samples=num_samples)
                # Analyze molecules
                analysis = analyze_molecules(generated_smiles, training_smiles_set)
                # Store results in session state
                st.session_state.generated_smiles = generated_smiles
                st.session_state.analysis = analysis
                st.session_state.vae_generated = True

            # Display summary
            st.success("✅ Molecule generation completed!")
            st.subheader("Summary of Generated Molecules")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Generated", analysis['total'])
            col2.metric("Valid Molecules", f"{analysis['valid']} ({(analysis['valid']/analysis['total'])*100:.2f}%)")
            col3.metric("Unique Molecules", f"{analysis['unique']} ({(analysis['unique']/analysis['total'])*100:.2f}%)")
            col4.metric("Corrected Molecules", f"{analysis['corrected']} ({(analysis['corrected']/analysis['total'])*100:.2f}%)")

            col1, col2 = st.columns(2)
            col1.metric("Novel Molecules", f"{analysis['novel']} ({(analysis['novel']/analysis['total'])*100:.2f}%)")
            col2.metric("Invalid Molecules", f"{analysis['invalid']} ({(analysis['invalid']/analysis['total'])*100:.2f}%)")

            # Display properties
            if analysis['valid_properties'] or analysis['novel_properties']:
                st.subheader("Properties of Generated Molecules")

                tab1, tab2 = st.tabs(["✅ Valid Molecules", "🌟 Novel Molecules"])
                with tab1:
                    if analysis['valid_properties']:
                        df_valid = pd.DataFrame(analysis['valid_properties'])
                        st.dataframe(df_valid)
                        # Visualize valid molecules (limit to 9 for performance)
                        st.subheader("Sample Valid Molecules")
                        mol_image_valid = visualize_molecules([prop['smiles'] for prop in analysis['valid_properties']], n=9)
                        if mol_image_valid:
                            st.image(mol_image_valid)
                        else:
                            st.write("No valid molecules to display.")
                    else:
                        st.write("No valid molecules found.")

                with tab2:
                    if analysis['novel_properties']:
                        df_novel = pd.DataFrame(analysis['novel_properties'])
                        st.dataframe(df_novel)
                        # Visualize novel molecules (limit to 9 for performance)
                        st.subheader("Sample Novel Molecules")
                        mol_image_novel = visualize_molecules([prop['smiles'] for prop in analysis['novel_properties']], n=9)
                        if mol_image_novel:
                            st.image(mol_image_novel)
                        else:
                            st.write("No novel molecules to display.")
                    else:
                        st.write("No novel molecules found.")

                # Property distributions
                st.subheader("Property Distributions")
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                if analysis['valid_properties']:
                    sns.histplot(df_valid['MolWt'], bins=20, ax=axs[0, 0], kde=True, color='skyblue', label='Valid')
                if analysis['novel_properties']:
                    sns.histplot(df_novel['MolWt'], bins=20, ax=axs[0, 0], kde=True, color='orange', label='Novel')
                axs[0, 0].set_title('Molecular Weight Distribution')
                axs[0, 0].legend()

                if analysis['valid_properties']:
                    sns.histplot(df_valid['LogP'], bins=20, ax=axs[0, 1], kde=True, color='skyblue', label='Valid')
                if analysis['novel_properties']:
                    sns.histplot(df_novel['LogP'], bins=20, ax=axs[0, 1], kde=True, color='orange', label='Novel')
                axs[0, 1].set_title('LogP Distribution')
                axs[0, 1].legend()

                if analysis['valid_properties']:
                    sns.histplot(df_valid['NumHDonors'], bins=range(0, max(df_valid['NumHDonors'].max(), 
                                                                         df_novel['NumHDonors'].max()) + 2), 
                                ax=axs[1, 0], kde=False, color='skyblue', label='Valid')
                if analysis['novel_properties']:
                    sns.histplot(df_novel['NumHDonors'], bins=range(0, max(df_valid['NumHDonors'].max(), 
                                                                         df_novel['NumHDonors'].max()) + 2), 
                                ax=axs[1, 0], kde=False, color='orange', label='Novel')
                axs[1, 0].set_title('Number of H Donors')
                axs[1, 0].legend()

                if analysis['valid_properties']:
                    sns.histplot(df_valid['NumHAcceptors'], bins=range(0, max(df_valid['NumHAcceptors'].max(), 
                                                                            df_novel['NumHAcceptors'].max()) + 2), 
                                ax=axs[1, 1], kde=False, color='skyblue', label='Valid')
                if analysis['novel_properties']:
                    sns.histplot(df_novel['NumHAcceptors'], bins=range(0, max(df_valid['NumHAcceptors'].max(), 
                                                                            df_novel['NumHAcceptors'].max()) + 2), 
                                ax=axs[1, 1], kde=False, color='orange', label='Novel')
                axs[1, 1].set_title('Number of H Acceptors')
                axs[1, 1].legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Download options
                csv_valid = df_valid.to_csv(index=False).encode('utf-8')
                csv_novel = df_novel.to_csv(index=False).encode('utf-8')
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Valid Molecules CSV",
                        data=csv_valid,
                        file_name='valid_molecules.csv',
                        mime='text/csv'
                    )
                with col2:
                    st.download_button(
                        label="💾 Download Novel Molecules CSV",
                        data=csv_novel,
                        file_name='novel_molecules.csv',
                        mime='text/csv'
                    )
            else:
                st.warning("No valid or novel molecules were generated.")

    elif model_option == "Predict Property (GNN)":
        # Load GNN models
        with st.spinner("Loading GNN models..."):
            gcn_model, gin_model = load_gnn_models(device)
            # Load normalization parameters
            data_mean, data_std = load_data_norm(device)

        # GNN Model selection
        gnn_model_option = st.sidebar.selectbox("Choose a GNN model", ("GCN", "GIN"))

        st.subheader("🔍 Predict Molecular Property using GNN")
        st.markdown("""
        Input a SMILES string to predict the dipole moment using the selected GNN model.
        """)

        # User inputs a SMILES string
        user_smiles = st.text_input("Enter a SMILES string for property prediction:", "")

        if user_smiles:
            data = smiles_to_data(user_smiles)
            if data:
                data = data.to(device)
                if gnn_model_option == "GCN":
                    prediction = gcn_model(data)
                    prediction = prediction.item()
                elif gnn_model_option == "GIN":
                    prediction = gin_model(data)
                    prediction = prediction.item()
                # Unnormalize the prediction
                prediction = prediction * data_std.item() + data_mean.item()
                st.success(f"**Predicted Dipole Moment ({gnn_model_option}):** {prediction:.4f}")
                # Display molecule
                mol = Chem.MolFromSmiles(user_smiles)
                if mol:
                    st.subheader("Molecular Structure")
                    st.image(Draw.MolToImage(mol, size=(300, 300)))
            else:
                st.error("❌ Invalid SMILES string.")

        st.markdown("---")
        st.markdown("### Or select a molecule from the generated molecules (if any).")

        # Check if molecules have been generated
        if st.session_state.vae_generated and st.session_state.analysis is not None:
            # Combine valid and novel properties
            all_properties = st.session_state.analysis['valid_properties'] + st.session_state.analysis['novel_properties']
            if all_properties:
                smiles_options = [prop['smiles'] for prop in all_properties]
                selected_smiles = st.selectbox("Select a molecule", smiles_options)
                if selected_smiles:
                    data = smiles_to_data(selected_smiles)
                    if data:
                        data = data.to(device)
                        if gnn_model_option == "GCN":
                            prediction = gcn_model(data)
                            prediction = prediction.item()
                        elif gnn_model_option == "GIN":
                            prediction = gin_model(data)
                            prediction = prediction.item()
                        # Unnormalize the prediction
                        prediction = prediction * data_std.item() + data_mean.item()
                        st.success(f"**Predicted Dipole Moment ({gnn_model_option}):** {prediction:.4f}")
                        # Display molecule
                        mol = Chem.MolFromSmiles(selected_smiles)
                        if mol:
                            st.subheader("Molecular Structure")
                            st.image(Draw.MolToImage(mol, size=(300, 300)))
                    else:
                        st.error("❌ Invalid SMILES string.")
            else:
                st.info("🔍 No valid or novel molecules available.")
        else:
            st.info("🔍 No generated molecules available. Generate molecules using the VAE first.")

    # About section
    st.sidebar.title("ℹ️ About")
    st.sidebar.info("""
    **Molecule Generator and Property Predictor App**

    This app uses a Variational Autoencoder (VAE) model and Graph Neural Networks (GNNs) to generate novel molecular structures and predict molecular properties.

    - **Developed by**: Arjun, Kaustubh, and Nachiket
    - **Hugging Face Repository**: https://huggingface.co/spaces/Raykarr/SMILES_Generation_and_Prediction
    """)

    # Hide Streamlit footer and header
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
