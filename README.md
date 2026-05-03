We introduce AGRec, a novel and extremely lightweight framework for recommendation. Unlike LLM-based models that require massive parameters, AGRec achieves superior performance with only 0.5M parameters by employing an adaptive granularity strategy and a decoupled inference mechanism.

Our experiments were conducted using PyTorch 2.9.1 on an NVIDIA RTX 5090 GPU.
For detailed environment settings (CUDA 12.0+)

Note: Large data files and model weights are managed separately.Some data files and model weights exceed GitHub's size limit.
Weights: The pre-trained model for Steam is available as step9_steam_agrec_pro.pth.
Datasets: Raw data (e.g., amazon_baby_inter) should be placed in the root or specified data directory.

We provide scripts to process interaction data and metadata:
# To process interaction data
python process_data.py 
# To process metadata
python process_meta.py

To train or run the AGRec model:python run_model.py

To run baseline comparisons or evaluate the model:
# Run baselines (e.g., BPR)
python run_baseline_BPR.py
# Performance testing
python test_agrec.py
# Statistical significance tests
python agrec_statistical_tests.py
# Ablation studies
python ablation.py
