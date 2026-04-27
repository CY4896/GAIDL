# GAIDL:  GenAI-Enhanced Deep Learning

This repository contains the implementation of GAIDL, a novel deep learning method for financial distress prediction.

## 📋 Requirements

To run this code, you need access to the following resources:

### Hardware
- **GPU**: NVIDIA GPU with CUDA support is recommended (e.g., RTX 3090 or A100).
- **RAM**: 16GB or higher system RAM is recommended.

### Software
- Python 3.8+
- PyTorch 1.10+
- Pandas, NumPy, Scikit-learn
- `rtdl` (Research on Tabular Deep Learning)

## 📂 Dataset

1.  **Financial  Indicators and Rating Ratios**
2.  **Text Embeddings**:
    -   **Current Reports**: FinBERT embeddings of 8-K reports 
    -   **AI-Generated Analysis Contents**: FinBERT embeddings of AI-generated analysis contents
3.  **Company Lists**: Python modules containing index mappings for text slicing (current reports & AI-generated analysis contents).

## 🚀 How to Run

1.  **Execution**:
    Run the main script:
    ```bash
    python main.py
    ```

2.  **Process**:
    -   Data Loading: the script loads financial and textual (embedding) data, using the train dataset as the training set and the test dataset as the testing set.
    -   Model Training: GAIDL model is trained using the train dataset.
    -   Model Evaluation: model performance is evaluated on the test dataset using Bootstrap sampling (50 times).

## 📊 Outcomes

Upon successful execution, the results will be saved in the `Results_bootstrap` directory:
-   **`Result1-50bootstrap_GAIDL`**: Results of 50 performance estimates for each metric.
-   **`Result2-bootstrap_mean_std_GAIDL`**: Summary of performance metrics (Mean ± Std Dev).
    -   **AUC**: Area Under the ROC Curve.
    -   **KS**: Kolmogorov-Smirnov statistic.
    -   **F1**: F1 Score.



