# AGRec: Adaptive Granularity Recommendation with Decoupled Inference

Official implementation of the paper **"AGRec: Adaptive Granularity Recommendation with Decoupled Inference"**.

## 🚀 Introduction
We introduce **AGRec**, a novel and extremely lightweight framework for recommendation. Unlike LLM-based models that require massive parameters, AGRec achieves superior performance with only **0.5M** parameters by employing an adaptive granularity strategy and a decoupled inference mechanism.

## 📈 Main Results
AGRec consistently outperforms strong baselines (such as RecGPT and LLM2Rec) across multiple public datasets.

| Model | Parameters | Steam | Musical | Yelp | Baby |

## 🛠️ Environment Setup
Our experiments were conducted using **PyTorch 2.9.1** on an **NVIDIA RTX 5090 GPU**.
```bash
pip install -r requirements.txt
```
📂 Datasets & Weights
Due to file size limits, large assets are managed as follows:

Weights: The pre-trained model for Steam is available as step9_steam_agrec_pro.pth.

Datasets: Raw data (e.g., amazon_baby_inter) should be placed in the root or ./data/ directory.
🏃 Usage Guide
1. Data Preprocessing
Use the following scripts to prepare your data and metadata:
Task,Command
Interaction Data,python process_data.py
Metadata,python process_meta.py
2. Model Training & Evaluation
To run the core AGRec framework:
```bash
python run_model.py
```
[!IMPORTANT]If you are running in a custom environment, ensure the following parameters are correctly set in your configuration:output_dir: Path to save checkpoints (seeds and samples are auto-appended).orth_loss: Weight for the orthogonality loss $\mathcal{L}_{orth}$.

### 3. Benchmarking & Analysis
Scripts for reproducibility and further analysis:

*   **Baselines:** 
    `python run_baseline_BPR.py`
*   **Performance Testing:** 
    `python test_agrec.py`
*   **Statistical Significance:** 
    `python agrec_statistical_tests.py`
*   **Ablation Studies:** 
    `python ablation.py`
