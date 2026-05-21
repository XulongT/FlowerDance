# FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.21029)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?logo=githubpages&logoColor=white)](https://sun-happy-ykx.github.io/FlowerDance/)
[![Training Code](https://img.shields.io/badge/Training-Code-4169e1?logo=github&logoColor=white)](#code)

![FlowerDance teaser](./flowerteaser.png)

> **Abstract**: Music-to-dance generation translates auditory signals into expressive human motion, yet existing approaches still struggle to balance refined 3D motion quality with strict inference budgets. FlowerDance is designed for both physically plausible, artistically expressive motion and efficient generation in speed and memory usage.
>
> FlowerDance combines MeanFlow with Physical Consistency Constraints for high-quality few-step sampling, and uses a lightweight non-autoregressive BiMamba backbone with Channel-Level Fusion for long-horizon music-to-dance synthesis. It also supports motion editing through time-decayed soft masking, enabling users to refine generated dance sequences interactively.

✨ Training code release! ✨

---

# Code

## Set up the Environment

To set up the necessary environment for running this project, follow the steps below:

1. **Create a new conda environment**

   ```bash
   conda create -n Flower_env python=3.10
   conda activate Flower_env
   ```

2. **Install PyTorch (CUDA 12.8)**

   ```
   pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   ```
   
3. **Install remaining dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Download Resources

- Download the **Preprocessed feature** from [Google Drive](https://drive.google.com/file/d/1UfCsOYMRsJAsH1LOxrJg8X5o3MRAwM2s/view?usp=sharing) and place them into `./data/` folder.
- Download the **Checkpoints for evaluation** and place them into the `./runs/` folder:  
  [Download Link](https://drive.google.com/file/d/1zZs_sXJToD5UzOA_m_DEoC0M79rEnEkg/view?usp=sharing)

---

## Directory Structure

After downloading the necessary files, ensure the directory structure follows the pattern below:

```
FlowerDance/
    │                
    ├── data/                 
    ├── dataset/             
    ├── model/                               
    ├── runs/  
    ├── requirements.txt
    ├── args.py  
    ├── EDGE.py
    ├── inpaint.py
    ├── test.py
    └── vis.py     
```
---

## Training

```bash
export WANDB_MODE=offline
accelerate launch train.py --batch_size 128  --epochs 4000 --feature_type baseline
```
---

## Evaluation

### Evaluate the Model

To evaluate the our model’s performance:

```bash
python test.py --batch_size 128
```


---

# Citation

```bibtex
@article{yang2025flowerdance,
  title={FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation},
  author={Kaixing Yang and Xulong Tang and Ziqiao Peng and Xiangyue Zhang and Puwei Wang and Jun He and Hongyan Liu},
  journal={arXiv preprint arXiv:2511.21029},
  year={2025}
}
```
