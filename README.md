# Automated Spectrogram-Based Machine Sound Analysis using Function-based Generated Noise and Parameter-Efficient Fine-Tuning Vision-Language Model

This project aims to analyze machine sounds by processing their spectrograms. It leverages two key innovations:

1. **Function-based Generated Noise:**  
   Noise is generated using a function-based approach to augment the spectrogram data, improving the robustness of the analysis.
   
2. **Parameter-Efficient Fine-Tuning (PEFT) for Vision-Language Models:**  
   Instead of fine-tuning all model parameters, only a subset is adjusted. This significantly reduces computational overhead and resource usage while maintaining performance on the spectrogram-based sound analysis task.

### Environment Setup 

Select one **installation method**:

##### 1. Using conda (with environment.yml)

```bash
conda env create -f environment.yml
conda activate function_based
```

##### 2. Using conda (with requirement.txt)

2.1 **Create a New Conda Environment**
    
```bash
conda create -n function_based python=3.11 -y
conda activate function_based
```

2.2 **Install PyTorch with NVIDIA CUDA Support**
    
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### How to start?

```bash
cd function_based/peft/no_based
python main.py
```

### Directory Tree

```bash
├── environment.yml
├── function_based
│   ├── noise_pipeline
│   ├── peft
│   │   ├── no_based
│   │   │   ├── logs
│   │   │   ├── augmentation 
│   │   │   ├── config.py                       -> can change variable, params...                  
│   │   │   ├── load_model.py                   
│   │   │   ├── main.py                         <- just start this
│   │   │   ├── peft_model_compare_table.xlsx   -> can see result here
├── README.md
└── requirements.txt
```

- If you want to create a new model, simply copy and paste the no_based folder within the same directory.
- If need more detail config of noise_pipeline..  you can edit `noise_pipeline/constants.py` or `noise_pipeline/shape_params.py`



---


todo:
- [ ] make grid search make. this make robust noise_pipeline. (Only random make be unstable?)
 # function_based_for_tango
# function_based_for_tango
