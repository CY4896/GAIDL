# config.py
import torch
import os


class Config:
    # Random Seed
    SEED = 2

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Base Directory (Modify this if needed)
    # BASE_DIR = "/Users/smuhn/PycharmProjects/AAA-MISQ第二轮返修实验"

    # --- Paths for 2022 (Train) ---
    FIN_DATA_2022 = "GAIDL_DATA/Findata-nasdaq2022.csv"
    RATIO_DATA_2022 = "GAIDL_DATA/nasdaq-ratio-2022.csv"
    TEXT_8K_2022 = "GAIDL_DATA/Finbert_8k_nasdaq_2022.csv"
    TEXT_GPT_2022 = "GAIDL_DATA/Finbert_gpt4omini_nasdaq_2022.csv"

    # --- Paths for 2023 (Test) ---
    FIN_DATA_2023 = "GAIDL_DATA/Findata-nasdaq2023.csv"
    RATIO_DATA_2023 = "GAIDL_DATA/nasdaq-ratio-2023.csv"
    TEXT_8K_2023 = "GAIDL_DATA/Finbert_8k_nasdaq_2023.csv"
    TEXT_GPT_2023 = "GAIDL_DATA/Finbert_gpt4omini_nasdaq_2023.csv"

    # --- Checkpoints ---
    MODEL_SAVE_PATH = 'GAIDL-nasdaq.pt'
    RESULT_DIR = "Results_bootstrap-nasdaq"

    # --- Hyperparameters ---
    MAX_LEN = 50
    BATCH_SIZE = 64
    EPOCHS = 100
    LR_MAIN = 0.001
    LR_MINE = 0.0005
    LAMBDA_MI = 0.001
    PATIENCE = 5
    ALPHA = 0.2
    GAMMA = 2
