# Deep Generative Models Enabled Label-free Segmentation for Electron Microscopy Images of Supported Nanoparticle

This repository contains scripts for nanoparticle image segmentation, mask generation, and electron microscopy image analysis. For more details on the methodology, refer to the associated [arXiv paper](https://arxiv.org/abs/2407.19544).

## Repository Structure

- **01_segmentation_predict_training**: 
  Tools for training deep learning models for segmentation and predicting masks on new images.

- **02_mask_generation**: 
  Scripts to generate synthetic nanoparticle masks with varying sizes and shapes and extract key parameters like particle size and ellipticity.

- **03_image_generate**: 
  Contains image generation and evaluation scripts, including Fréchet Inception Distance (FID), Peak Signal-to-Noise Ratio (PSNR), Inception Score (IS), Kernel Maximum Mean Discrepancy (MMD), and other quality assessment methods.

- **04_in_situ_analysis**: 
  Tools for processing and segmenting electron microscopy images (DM4 format), extracting particle properties, and performing statistical analysis on particle size distributions.

## Usage

1. **Train and predict segmentation models**: Located in the `01_segmentation_predict_training` folder.
2. **Generate and analyze synthetic masks**: Available in `02_mask_generation`.
3. **Generate and evaluate images**: Use the scripts in `03_image_generate` to train, generate, and assess image quality.
4. **Analyze DM4 microscopy data**: Use the tools in `04_in_situ_analysis` to analyze HAADF-STEM images of supported nanoparticles in real time.

## Installation

Install required packages using:
```bash
pip install -r requirements.txt
```
## Citation
If you find our code or data useful in your research, please cite our paper:
```
@misc{yuan2024deepgenerativemodelsassistedautomated,
      title={Deep Generative Models-Assisted Automated Labeling for Electron Microscopy Images Segmentation}, 
      author={Wenhao Yuan and Bingqing Yao and Shengdong Tan and Fengqi You and Qian He},
      year={2024},
      eprint={2407.19544},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2407.19544}, 
}
```
Copyright | 2024 Qian's Lab@NUS 

