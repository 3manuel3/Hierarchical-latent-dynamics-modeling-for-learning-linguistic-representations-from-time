"""
run_dcl_raw_level0.py

Primo livello HLDL: DCL su chunk raw da 0.020 s (320 campioni) di LibriSpeech dev-clean.

- Ogni time step = finestra di 0.020 s (16 kHz -> 320 samples).
- Encoder = MLP di DCL (input_dim = 320).
- Dynamics = LinearDynamicsModel (dinamica di primo ordine nello spazio latente).
- Questo livello corrisponde alla scala temporale più “locale” di HLDL;
  livelli gerarchici superiori (fonemi, sillabe, parole) verranno costruiti in seguito
  a partire dalle embedding ottenute qui.



Prerequisiti:
    - repo DCL installato (es. in env conda sul server):
          pip install -e path/to/dcl-main
    - PyTorch (+ CUDA sul server se disponibile).

Suggerito: lanciare su server con:
    CUDA_VISIBLE_DEVICES=0 python run_dcl_raw_level0.py
"""

from pathlib import Path

import torch

from dcl.datasets.timeseries import TensorDataset
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS, MSESwitchingModel
from dcl.models.encoder import MLP
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.solver.optimizer import DCLAdamOptimizer
from dcl.criterions.contrastive import MseInfoNCE


# --------------------------- CONFIG ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "dataset" / "dev-clean-raw20"

SEED = 42

# Se il server ha abbastanza RAM/GPU, possiamo usare tutti i dati.
# Puoi ridurre questi limiti se serve.
MAX_UTTERANCES = None       # None = usa tutte le utterances trovate
MAX_FRAMES_PER_UTT = None   # None = usa tutti i frame per utterance

BATCH_SIZE = 512            # puoi aumentare se la GPU lo regge
LATENT_DIM = 64             # dimensione spaziale latente per livello 0
ENC_HIDDEN_DIM = 512
ENC_LAYERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ COSTRUZIONE TENSORDATASET ----------------------

def build_raw_chunk_dataset(root: Path) -> TensorDataset:
    """
    Legge i .pt [num_frames, FRAME_SAMPLES] da root,
    opzionalmente li sottocampiona, e concatena tutti i frame:

        data: Tensor [T_tot, FRAME_SAMPLES]

    Ritorna un TensorDataset DCL (livello 0).
    """
    if not root.exists():
        raise FileNotFoundError(f"Cartella raw non trovata: {root}")

    pt_files = sorted(root.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"Nessun .pt trovato sotto {root}")

    if MAX_UTTERANCES is not None:
        pt_files = pt_files[:MAX_UTTERANCES]

    print(f"Utterances usate: {len(pt_files)}")

    frames_list = []
    frame_dim = None

    for i, path in enumerate(pt_files, start=1):
        x = torch.load(path, map_location="cpu")   # [num_frames, FRAME_SAMPLES]
        if x.ndim != 2:
            raise ValueError(f"{path} non è 2D, shape={x.shape}")

        num_frames, dim = x.shape
        frame_dim = dim if frame_dim is None else frame_dim

        if MAX_FRAMES_PER_UTT is not None and num_frames > MAX_FRAMES_PER_UTT:
            x = x[:MAX_FRAMES_PER_UTT]

        frames_list.append(x)

        if i % 100 == 0 or i == len(pt_files):
            print(
                f"[{i}/{len(pt_files)}] {path.name} "
                f"frames_usati={x.shape[0]}, frame_dim={x.shape[1]}"
            )

    data = torch.cat(frames_list, dim=0)  # [T_tot, FRAME_SAMPLES]
    print(f"Tensor concatenato data shape: {tuple(data.shape)}  (T_tot, FRAME_SAMPLES={frame_dim})")

    dataset = TensorDataset(data=data)
    print(f"dataset.observed_dim = {dataset.observed_dim}")
    return dataset


# ---------------------- PIPELINE DCL LIVELLO 0 ----------------------

def build_dcl_pipeline_level0(dataset: TensorDataset):
    """
    Costruisce loader, encoder MLP e dynamics per il livello 0 (0.020 s).
    """
    # 1) Data loader contrastivo
    loader = DiscreteTimeContrastiveDataLoader(batch_size=BATCH_SIZE, seed=SEED)
    loader.lazy_init(dataset)

    # 2) Encoder MLP su vettori raw di lunghezza FRAME_SAMPLES
    encoder = MLP(
        input_dim=dataset.observed_dim,   # ~320
        output_dim=LATENT_DIM,
        hidden_dim=ENC_HIDDEN_DIM,
        num_layers=ENC_LAYERS,
    ).to(DEVICE)

    # 3) Modello di dinamica (puoi cambiare in SLDS se vuoi switching dynamics)
    linear_dynamics = LinearDynamicsModel(dim=LATENT_DIM).to(DEVICE)

    # Opzione alternativa: SLDS (commentata per ora)
    num_modes = 5
    slds_dynamics = GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            dim=LATENT_DIM,
            num_systems=num_modes,
        ),
        switching_model=MSESwitchingModel(num_modes=num_modes),
    ).to(DEVICE)

    dynamics_model = linear_dynamics  # o slds_dynamics

    # 4) Solver DCL
    solver = DynamicsContrastiveLearningSolver(
        model=encoder,
        dynamics_model=dynamics_model,
        optimizer=DCLAdamOptimizer(
            encoder_learning_rate=3e-4,
            dynamics_learning_rate=3e-3,
        ),
        criterion=MseInfoNCE(
            temperature=1.0,
            infonce_type="infonce_full_denominator",
        ),
        device=DEVICE,  # se l'API DCL espone questo argomento
    )

    return loader, solver


# ------------------------------ MAIN --------------------------------

def main():
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Costruisco TensorDataset livello 0 da {RAW_ROOT} ...")
    dataset = build_raw_chunk_dataset(RAW_ROOT)

    print("Inizializzo DCL (loader + solver, livello 0)...")
    loader, solver = build_dcl_pipeline_level0(dataset)

    print("Inizio training DCL livello 0 (chunk 0.020 s)...")
    solver.fit(loader)

    print("Calcolo predizioni (embeddings + dynamics) per livello 0...")
    predictions = solver.predictions(loader)
    embeddings = predictions.embeddings     # [T_tot, LATENT_DIM]
    dynamics_predictions = predictions.dynamics

    print(f"Embeddings shape (level 0): {tuple(embeddings.shape)}")
    print(f"Dynamics predictions shape (level 0): {tuple(dynamics_predictions.shape)}")

    # (opzionale) salva le embedding per usarle nei livelli gerarchici successivi di HLDL
    out_dir = PROJECT_ROOT / "output" / "level0"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.cpu(), out_dir / "embeddings_level0.pt")
    torch.save(dynamics_predictions.cpu(), out_dir / "dynamics_level0.pt")
    print(f"Salvate embedding e dynamics in {out_dir}")


if __name__ == "__main__":
    main()
