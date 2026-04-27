# main.py
import os
import csv
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config
from data_processor import TrainDataset, ValidationDataset, load_all_text_data
from models import Model
from utils import seed_it, FocalLoss, EarlyStopping, bootstrap_evaluate


def main():
    # 1. Device check
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA device index: {current_device}")
        print(f"Current CUDA device name: {device_name}")
    else:
        print("CUDA is not available.")

    seed_it(Config.SEED)
    print(f'Using device: {Config.DEVICE}')

    # 2. Financial Data Processing (Logic from Segment 1)
    scaler = StandardScaler()

    # 2022 Data
    cwtz2022 = pd.read_csv(Config.FIN_DATA_2022)
    # y_labels2022 = cwtz2022.label.values (Not used directly here, but read)
    print(cwtz2022.head())
    X_cw2022 = cwtz2022.iloc[:, 1:13].values
    X_cw2022 = scaler.fit_transform(X_cw2022)
    X_cwgpt2022 = pd.read_csv(Config.RATIO_DATA_2022)
    X_cwgpt2022 = X_cwgpt2022.iloc[:, 1:3].values
    X_cw2022 = np.hstack((X_cw2022, X_cwgpt2022))

    # 2023 Data
    cwtz2023 = pd.read_csv(Config.FIN_DATA_2023)
    # y_labels2023 = cwtz2023.label.values
    print(cwtz2023.head())
    X_cw2023 = cwtz2023.iloc[:, 1:13].values
    X_cw2023 = scaler.transform(X_cw2023)
    X_cwgpt2023 = pd.read_csv(Config.RATIO_DATA_2023)
    X_cwgpt2023 = X_cwgpt2023.iloc[:, 1:3].values
    X_cw2023 = np.hstack((X_cw2023, X_cwgpt2023))

    # 3. Load Text Data (Calls logic from Segment 2 & 3)
    text2022, gpt2022, y_labels2022, text2023, gpt2023, y_labels2023 = load_all_text_data()

    y_train = y_labels2022
    y_test = y_labels2023

    # 4. Data Assignment & Split
    # X_train1, X_test1 -> Financial
    # X_train2, X_test2 -> Text 8k
    # X_train4, X_test4 -> GPT
    X_train1, X_test1 = X_cw2022, X_cw2023
    X_train2, X_test2 = text2022, text2023
    X_train4, X_test4 = gpt2022, gpt2023

    X_train2 = np.array(X_train2)
    X_train4 = np.array(X_train4)

    train_indices, valid_indices = train_test_split(np.arange(len(y_train)), test_size=0.2, random_state=42)
    train_indices = np.array(train_indices, dtype=int)
    valid_indices = np.array(valid_indices, dtype=int)

    X_train1_split, X_valid1 = X_train1[train_indices], X_train1[valid_indices]
    X_train2_split, X_valid2 = X_train2[train_indices], X_train2[valid_indices]
    X_train4_split, X_valid4 = X_train4[train_indices], X_train4[valid_indices]
    y_train_split, y_valid = y_train[train_indices], y_train[valid_indices]

    # Convert labels to DataFrame as per original code
    y_train_df = pd.DataFrame(y_train_split)

    # 5. Dataset & Loader
    trainset = TrainDataset(X_train1_split, X_train2_split, X_train4_split, y_train_df)
    train_loader = DataLoader(trainset, batch_size=Config.BATCH_SIZE)

    valid_dataset = ValidationDataset(X_valid1, X_valid2, X_valid4, y_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 6. Model & Optimizer
    criterion = FocalLoss(alpha=Config.ALPHA, gamma=Config.GAMMA, logits=False, reduction='mean')
    ourmodel = Model()

    optimizer = torch.optim.AdamW(ourmodel.Embedding.parameters(), lr=Config.LR_MAIN)
    mine_optimizer = torch.optim.AdamW(ourmodel.mine.parameters(), lr=Config.LR_MINE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=Config.PATIENCE)
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True, path=Config.MODEL_SAVE_PATH)

    # 7. Training Loop
    ourmodel.train()
    for epoch in range(Config.EPOCHS):
        print("Current learning rate: ", optimizer.param_groups[0]['lr'])
        for step, (x_1, x_2, x_3, y) in enumerate(train_loader):
            mine_optimizer.zero_grad()

            x_data = [x_1, x_2, x_3]
            y = y.reshape(-1, 1).to(torch.float64)

            pred, total_loss, total_train = ourmodel(x_data)
            mi_loss = total_train
            mi_loss.backward()
            mine_optimizer.step()

            pred, total_loss, total_train = ourmodel(x_data)
            pred = pred.to(torch.float64)
            bce_loss = criterion(pred, y)
            mainnetwork_loss = bce_loss + Config.LAMBDA_MI * total_loss

            optimizer.zero_grad()
            mainnetwork_loss.backward()
            optimizer.step()

        ourmodel.eval()
        total_valid_loss = 0
        total_valid_mi_loss = 0

        for step, (x_1, x_2, x_3, y) in enumerate(valid_loader):
            x_data = [x_1, x_2, x_3]
            y = y.reshape(-1, 1).to(torch.float64)

            pred, total_loss_valid, total_valid = ourmodel(x_data)
            pred = pred.to(torch.float64)
            mi_loss_valid = total_valid

            bce_loss_valid = criterion(pred, y)
            mainnetwork_loss = bce_loss_valid + Config.LAMBDA_MI * total_loss_valid

            total_valid_loss += mainnetwork_loss.item()
            total_valid_mi_loss += mi_loss_valid.item()

        scheduler.step(total_valid_loss / len(valid_loader))

        print(f'Epoch {epoch}, Validation Total Loss: {total_valid_loss}, Validation MI Loss: {total_valid_mi_loss}')

        early_stopping(total_valid_loss, ourmodel)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 8. Evaluation
    ourmodel.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    # Prepare test data manually as per original 'def test()'
    X_test_cw = torch.from_numpy(X_test1.astype(np.float32))
    X_test_tfeature = torch.Tensor(X_test2)
    X_test_gpt = torch.from_numpy(X_test4.astype(np.float32))
    y_test_df = pd.DataFrame(y_test)
    test_labels = y_test_df.values.astype(np.float64)
    test_labels = torch.from_numpy(test_labels.reshape(-1))
    testdata = [X_test_cw, X_test_tfeature, X_test_gpt, test_labels]

    # Create result dir
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    metric_csv_path = os.path.join(Config.RESULT_DIR, 'Result1-50bootstrap_GAIDL_nasdaq2023.csv')
    meanstd_csv_path = os.path.join(Config.RESULT_DIR, 'Result2-bootstrap_mean_std_GAIDL_nasdaq2023.csv')
    # preds_csv_path = os.path.join(Config.RESULT_DIR, 'Result15-all_labels_predi_GAIDL_nasdaq2023.csv')

    all_metrics = bootstrap_evaluate(ourmodel, testdata, n_bootstrap=50, save_pred_path=metric_csv_path)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metric_csv_path, index=False)

    with open(meanstd_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Mean (Standard Deviation)'])
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        for metric, name in [('AUC', 'AUC'), ('KS', 'KS'), ('F1_1', 'Risk F1')]:
            mean_value = mean_metrics[metric]
            std_value = std_metrics[metric]
            mean_std = f"{mean_value:.3f} ({std_value:.3f})"
            writer.writerow([name, mean_std])
            print(f"{name}: {mean_std}")

    # print(f"Results with mean and standard deviation have been saved to '{meanstd_csv_path}'")


if __name__ == "__main__":
    main()