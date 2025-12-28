# 🚗 Car Semantic Segmentation: From Baseline to Transformers

This project explores the evolution of architectures for multiclass semantic segmentation applied to the automotive industry. The work analyzes the transition from standard UNet models to Vision Transformers (MiT-B2), solving critical class imbalance problems.

## 🗝️ Features

- **Multi-Class Segmentation**: Background, Body, Wheels, Headlights, Windows.
- **Loss Strategy**: Combination of Focal Loss and Dice Loss (Kornia) with logarithmic class weighting.
- **Architectures**: Baseline Unet, Deep Unet, Attention Unet, ResNet-34 Unet, MiT-B2 Unet.
- **Out-of-Distribution Testing**: Robustness testing on real images (Alfa Giulia, Lamborghini Urus, Fiat Panda).

## 📊 Performance Highlights

| Model | Parameters | IoU (Global) | IoU (Headlights) | Status |
| :-- | :-- | :-- | :-- | :-- |
| **Baseline** | 1.9 M | 22.55% | 0% | **Failed** |
| **ResNet 34** | 24.4 M | 87.21% | 64.6% | **Top** |
| **Mit-b2** | 26.4 M | 86.56% | 62.099% | **SOTA** |

## 🖼️ Visual Results

Example of segmentation on images external to the dataset

| **Unet - ResNet 34** | **Unet - Mit-b2** |
| :-- | :-- |
| ![Comparison](/infer/unet_resnet34/cover_alfa-giulia.png) | ![Comparison](/infer/unet_mit/cover_alfa-giulia.png) |
| ![Comparison](/infer/unet_resnet34/cover_lamborghini-urus.png) | ![Comparison](/infer/unet_mit/cover_lamborghini-urus.png) |
| ![Comparison](/infer/unet_resnet34/cover_pandino-storico.png) | ![Comparison](/infer/unet_mit/cover_pandino-storico.png) |

## 📂 Project Structure

- **/architecture**: Definition of the 5 different Unets tested
- **/dataset**: Data preparation, preprocessing, Data Augmentation, Split and DataLoader
- **/engine**: Early Stopping Function, Metrics (IoU and Dice Score), Loop train-validate, Plot of graphs
- **/graph**: Training graphs, Comparison images
- **/infer**: Images external to the dataset and their results (masks and overlays), divided by model
- **/run**: Main scripts

## 🔚 Conclusions and References

The project demonstrates that Transfer Learning and proper loss weighting are critical for segmenting minority classes (Headlights < 1% of pixels). The MiT-B2 model proved to be the most robust in handling complex geometric details.

Dataset used: [Car segmentation ](https://www.kaggle.com/datasets/intelecai/car-segmentation "Kaggle") 
