# CIFAR-10 Custom CNN Implementation

## Model Architecture Summary

### Network Design
- **Total Parameters**: 184,824 (under 200k requirement)
- **Receptive Field**: 47x47 (exceeds minimum requirement of 44)
- **Model Size**: 0.71 MB
- **Memory Usage**: 35.18 MB (total estimated)

### Architecture Blocks
1. **C1 Block (Initial Features)**
   - Input → 64 channels (5x5 kernel) → 96 channels
   - RF: 5x5 → 7x7
   - Uses Depthwise Separable Convolution

2. **C2 Block (Dilated Features)**
   - 96 → 128 → 160 channels
   - RF: 15x15 → 19x19
   - Uses Dilated Convolution (dilation=2)

3. **C3 Block (Deep Features)**
   - 160 → 192 → 224 channels
   - RF: 35x35 → 39x39
   - Uses Dilated Convolution (dilation=4)

4. **C4 Block (Final Features)**
   - 224 → 256 channels
   - RF: 47x47
   - Strided Convolution (stride=2)

### Key Components
- Depthwise Separable Convolutions throughout
- Dilated Convolutions for RF expansion
- Global Average Pooling
- BatchNormalization and ReLU activations

## Training Results

### Performance Metrics
- **Best Test Accuracy**: 89.08%
- **Final Training Accuracy**: 89.65%
- **Training Time**: ~35-40 seconds per epoch
- **Total Epochs**: 25

### Training Progression
- **Early Stage (1-5 epochs)**
  - Started at 39.63% accuracy
  - Rapid improvement to 66.69%
  - Test accuracy reached 72.70%

- **Mid Stage (10-15 epochs)**
  - Training accuracy improved to 80.85%
  - Test accuracy crossed 85% barrier
  - Stable learning progression

- **Final Stage (20-25 epochs)**
  - Training accuracy reached 89.65%
  - Test accuracy peaked at 89.08%
  - Showed good generalization

### Key Achievements
- Exceeded target accuracy of 85%
- Maintained parameters under 200k limit
- Achieved required receptive field (>44)
- Good convergence with minimal overfitting

### Training Parameters
- Batch Size: 128
- Learning Rate: Started at 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Data Augmentation: HorizontalFlip, ShiftScaleRotate, CoarseDropout 