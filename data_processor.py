# data_processor.py
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from config import Config
# Assuming these files exist in your directory as per original code
from GAIDL_DATA.companylist_nasdaq import list2022, list2023



class TrainDataset(Dataset):
    def __init__(self, X_train_cw, X_train_tfeature, X_train_gpt, y_train):
        super(TrainDataset, self).__init__()
        X_train_cw = torch.from_numpy(X_train_cw.astype(np.float32))
        X_train_tfeature = torch.Tensor(X_train_tfeature)
        X_train_gpt = torch.Tensor(X_train_gpt.astype(np.float32))
        y_train = pd.DataFrame(y_train)
        train_labels = y_train.values.astype(np.float64)
        train_labels = torch.from_numpy(train_labels.reshape(-1))

        self.len = y_train.shape[0]
        self.x_data = [X_train_cw, X_train_tfeature, X_train_gpt]
        self.y_data = train_labels

    def __getitem__(self, item):
        return self.x_data[0][item], self.x_data[1][item], self.x_data[2][item], self.y_data[item]

    def __len__(self):
        return self.len


class ValidationDataset(Dataset):
    def __init__(self, X_valid_cw, X_valid_tfeature, X_valid_gpt, y_valid):
        super(ValidationDataset, self).__init__()
        X_valid_cw = torch.from_numpy(X_valid_cw.astype(np.float32))
        X_valid_tfeature = torch.Tensor(X_valid_tfeature)
        X_valid_gpt = torch.Tensor(X_valid_gpt.astype(np.float32))
        y_valid = pd.DataFrame(y_valid)
        valid_labels = y_valid.values.astype(np.float64)
        valid_labels = torch.from_numpy(valid_labels.reshape(-1))

        self.len = y_valid.shape[0]
        self.x_data = [X_valid_cw, X_valid_tfeature, X_valid_gpt]
        self.y_data = valid_labels

    def __getitem__(self, item):
        return self.x_data[0][item], self.x_data[1][item], self.x_data[2][item], self.y_data[item]

    def __len__(self):
        return self.len


def process_text_segment(lsgg_path, gpt_path, company_list, fin_path, year_label):
    """
    Simulates the exact logic of dataloader_text_padding2022/2023.py
    """
    # Read financial data for print/check (as per original code)
    cwtz = pd.read_csv(fin_path)
    labels = cwtz.label.values
    print(cwtz.head())

    # Read text csvs
    lsgg_file = pd.read_csv(lsgg_path)
    gpt_file = pd.read_csv(gpt_path)

    data = np.array(lsgg_file)
    date = data.tolist()

    data1 = np.array(gpt_file)
    date1 = data1.tolist()

    text_list = []
    for i in company_list:
        text_list.append(date[i[0]:i[1]])
    print(f"text{year_label}:", len(text_list))

    gpt_list = []
    for i in company_list:
        gpt_list.append(date1[i[0]:i[1]])
    print(f"gpt{year_label}:", len(gpt_list))

    max_len = Config.MAX_LEN

    # Convert to tensor and truncate
    text_tensors = [
        torch.tensor(company_announcements, dtype=torch.float32)[:max_len]
        for company_announcements in text_list
    ]
    gpt_tensors = [
        torch.tensor(gpt_reply, dtype=torch.float32)[:max_len]
        for gpt_reply in gpt_list
    ]

    # Pad
    padded_text = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    padded_gpt = pad_sequence(gpt_tensors, batch_first=True, padding_value=0)

    text_out = np.array(padded_text)
    print(padded_text.shape)
    gpt_out = np.array(padded_gpt)
    print(padded_gpt.shape)
    y_labels = np.array(labels)

    return text_out, gpt_out, y_labels


def load_all_text_data():
    # Process 2022
    text2022, gpt2022, y_labels2022 = process_text_segment(
        Config.TEXT_8K_2022, Config.TEXT_GPT_2022, list2022, Config.FIN_DATA_2022, "2022"
    )
    # Process 2023
    text2023, gpt2023, y_labels2023 = process_text_segment(
        Config.TEXT_8K_2023, Config.TEXT_GPT_2023, list2023, Config.FIN_DATA_2023, "2023"
    )
    return text2022, gpt2022, y_labels2022, text2023, gpt2023, y_labels2023
