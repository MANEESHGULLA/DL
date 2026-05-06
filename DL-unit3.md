# Unit 3: Pre-trained CNN Models & CNN Visualization
## Deep Learning Mid Examination Study Guide

---

## Table of Contents
1. [Pre-trained Models - Overview](#overview)
2. [LeNet-5](#lenet-5)
3. [AlexNet](#alexnet)
4. [ZF-Net (Zeiler and Fergus Network)](#zf-net)
5. [VGGNet](#vggnet)
6. [GoogLeNet (Inception)](#googlenet)
7. [ResNet](#resnet)
8. [CNN Visualization Techniques](#cnn-visualization)

---

## Overview of Pre-trained Models

### Definition
A **pre-trained model** is a neural network that has been previously trained on a large dataset (such as ImageNet) and whose learned parameters (weights) are reused for solving a new but related problem.

### Why Use Pre-trained Models?
- **Transfer Learning**: Leverage learned features from large datasets
- **Reduced Training Time**: Start from trained weights instead of random initialization
- **Better Performance**: Models trained on millions of images generalize better
- **Resource Efficiency**: Avoid training massive models from scratch

### Two Approaches to Using Pre-trained Models:

#### 1. **Feature Extraction**
- Use the model as a **fixed feature extractor**
- Train **only the final layer** on your new dataset
- Best when: Limited data, quick training needed

#### 2. **Fine-Tuning**
- Train **some of the deeper layers** along with the new dataset
- Keep early layers frozen (low-level features are universal)
- Best when: Sufficient data, similar task domain

### Common Pre-trained Models:
- LeNet, AlexNet, VGGNet, ResNet
- GoogleNet (Inception), ZF-Net
- DenseNet, MobileNet, EfficientNet, Inception

---

## LeNet-5

### Historical Significance
- **First practical CNN** for real-world applications
- Developed by Yann LeCun in 1998
- Used for handwritten digit recognition (MNIST)
- Foundation for all modern CNNs

### LeNet-5 Architecture

#### **Input Layer**
- **Size**: 32×32 grayscale images
- MNIST images (28×28) are zero-padded to 32×32

#### **Convolutional Layer 1 (C1)**
- **Filters**: 6 filters of size 5×5
- **Output size**: 28×28×6
- **Activation**: Sigmoid/Tanh

#### **Average Pooling Layer (S2)**
- **Pool size**: 2×2
- **Stride**: 2
- **Output size**: 14×14×6

#### **Convolutional Layer 2 (C3)**
- **Filters**: 16 filters of size 5×5
- **Output size**: 10×10×16
- **Activation**: Sigmoid/Tanh

#### **Average Pooling Layer (S4)**
- **Pool size**: 2×2
- **Stride**: 2
- **Output size**: 5×5×16

#### **Flattening**
- Total neurons: 5×5×16 = 400

#### **Fully Connected Layer (F5)**
- **Neurons**: 120
- **Activation**: Tanh

#### **Fully Connected Layer (F6)**
- **Neurons**: 84
- **Activation**: Tanh

#### **Output Layer**
- **Neurons**: 10 (for digits 0-9)
- **Activation**: Softmax

### How Convolutional Layers Learn Complex Features (PYQ)

#### **Feature Hierarchy**:
1. **Layer C1 (First Conv Layer)**
   - Learns **low-level features**: edges, corners, blobs
   - Small 5×5 filters capture local patterns
   - Each filter specializes in detecting specific edge orientations

2. **Layer C3 (Second Conv Layer)**
   - Receives 14×14×6 input (combination of C1 features)
   - Learns **mid-level features**: textures, simple shapes
   - Combines outputs from multiple C1 filters
   - Can detect complex patterns by combining edge features

#### **How This Works**:
```
Input Image (32×32)
    ↓
C1 Layer: 6 filters detect edges/corners/blobs
    ↓
S2 Pooling: Downsample while preserving important features
    ↓
C3 Layer: 16 filters combine C1 features → detect shapes/textures
    ↓
Feature Maps: Early layers = simple edges
              Later layers = complex patterns
```

#### **Mathematical Insight**:
- **Local Connectivity**: Each neuron connects only to local region
- **Weight Sharing**: Same filter applied across entire image
- **Compositionality**: Complex features = compositions of simpler features

### LeNet-5 Code Implementation (Keras/TensorFlow)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Pad images from 28x28 to 32x32
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]])
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]])

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build LeNet-5 model
model = keras.Sequential([
    
    # C1 Convolution Layer
    layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        activation='tanh',
        input_shape=(32, 32, 1)
    ),
    
    # S2 Average Pooling
    layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=2
    ),
    
    # C3 Convolution Layer
    layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        activation='tanh'
    ),
    
    # S4 Average Pooling
    layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=2
    ),
    
    # C5 Convolution Layer
    layers.Conv2D(
        filters=120,
        kernel_size=(5, 5),
        activation='tanh'
    ),
    
    # Flatten Layer
    layers.Flatten(),
    
    # F6 Fully Connected Layer
    layers.Dense(
        84,
        activation='tanh'
    ),
    
    # Output Layer
    layers.Dense(
        10,
        activation='softmax'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict sample image
predictions = model.predict(x_test)

print("Predicted Digit:", predictions[0].argmax())
print("Actual Digit:", y_test[0].argmax())
```

---

## AlexNet

### Overview
- **Developers**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Year**: 2012
- **Achievement**: Won ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- **Impact**: Revolutionized deep learning; demonstrated power of deep CNNs on GPUs

### What AlexNet is Used For (PYQ)

#### **Primary Use Cases**:
1. **Image Classification**: Classify images into 1000 ImageNet categories
   - Cats, dogs, vehicles, objects, animals, scenes
   
2. **Feature Extraction**: Pre-trained on ImageNet for transfer learning
   - Extract learned features for downstream tasks
   - Object detection, segmentation, similarity search
   
3. **Benchmark Model**: Standard baseline for CNN performance
   - Compare new architectures against AlexNet
   - Establish state-of-the-art baselines

#### **Practical Applications**:
- Medical image analysis (disease detection)
- Autonomous vehicles (object recognition)
- Facial recognition systems
- Content-based image retrieval
- Product recommendation systems

### AlexNet Architecture

**Type**: Deep Convolutional Neural Network (8 layers)

#### **Layer Details**:

| Layer | Type | Parameters | Output Shape | Activation |
|-------|------|-----------|--------------|------------|
| Input | Image | - | (1, 3, 224, 224) | - |
| Conv1 | Conv2D | 96 filters, 11×11, stride 4 | (1, 96, 54, 54) | ReLU |
| Pool1 | MaxPool | 3×3, stride 2 | (1, 96, 26, 26) | - |
| Conv2 | Conv2D | 256 filters, 5×5, padding 'same' | (1, 256, 26, 26) | ReLU |
| Pool2 | MaxPool | 3×3, stride 2 | (1, 256, 12, 12) | - |
| Conv3 | Conv2D | 384 filters, 3×3, padding 'same' | (1, 384, 12, 12) | ReLU |
| Conv4 | Conv2D | 384 filters, 3×3, padding 'same' | (1, 384, 12, 12) | ReLU |
| Conv5 | Conv2D | 256 filters, 3×3, padding 'same' | (1, 256, 12, 12) | ReLU |
| Pool3 | MaxPool | 3×3, stride 2 | (1, 256, 5, 5) | - |
| Dense1 | FC | 4096 neurons | (1, 4096) | ReLU |
| Dropout1 | Dropout | 0.5 | (1, 4096) | - |
| Dense2 | FC | 4096 neurons | (1, 4096) | ReLU |
| Dropout2 | Dropout | 0.5 | (1, 4096) | - |
| Output | FC | 1000 neurons | (1, 1000) | Softmax |

### Output Dimension Calculation (For Reference)

**Batch Processing Output Shapes**:
- `conv0`: (1, 96, 54, 54)
- `pool0`: (1, 96, 26, 26)
- `conv1`: (1, 256, 26, 26)
- `pool1`: (1, 256, 12, 12)
- `conv2`: (1, 384, 12, 12)
- `conv3`: (1, 384, 12, 12)
- `conv4`: (1, 256, 12, 12)
- `pool2`: (1, 256, 5, 5)
- `dense0`: (1, 4096)
- `dropout0`: (1, 4096)
- `dense1`: (1, 4096)
- `dropout1`: (1, 4096)
- `output`: (1, 1000)

### Key Innovations of AlexNet

#### **1. ReLU Activation Function**
```
ReLU(x) = max(0, x)
```
- **Advantage over Sigmoid/Tanh**: 
  - Faster training (6× speedup)
  - Avoids vanishing gradient problem
  - Computationally efficient

#### **2. Dropout Regularization**
- **Rate**: 0.5 (50% neurons randomly dropped during training)
- **Purpose**: Prevent co-adaptation, reduce overfitting
- **Applied**: In fully connected layers (Dense1, Dense2)

#### **3. GPU Training**
- Trained on NVIDIA GPUs (2 GPUs in original paper)
- Made deep learning practical for large datasets
- Enabled 1000× speedup compared to CPU

#### **4. Data Augmentation**
- Random crops, horizontal flips
- Color jittering
- Increases effective training set size

### Differences Between LeNet and AlexNet

| Aspect | LeNet | AlexNet |
|--------|-------|---------|
| **Depth** | 5 layers | 8 layers |
| **Activation** | Sigmoid/Tanh | ReLU |
| **Regularization** | Minimal | Dropout (0.5) |
| **Dataset** | MNIST (28×28) | ImageNet (224×224) |
| **Classes** | 10 | 1000 |
| **Parameters** | ~60K | ~60M |
| **Training Speed** | CPU-friendly | GPU required |
| **Performance** | 99%+ accuracy | 63% Top-5 accuracy |

### AlexNet Code Implementation (Keras/TensorFlow)

```python
model = keras.Sequential([
    layers.Conv2D(96, kernel_size=(11, 11), strides=4, 
                  activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    
    layers.Conv2D(256, kernel_size=(5, 5), padding='same', 
                  activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    
    layers.Conv2D(384, kernel_size=(3, 3), padding='same', 
                  activation='relu'),
    layers.Conv2D(384, kernel_size=(3, 3), padding='same', 
                  activation='relu'),
    layers.Conv2D(256, kernel_size=(3, 3), padding='same', 
                  activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    
    layers.Flatten(),
    
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1000, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

---

## ZF-Net (Zeiler and Fergus Network)

### Overview
- **Developers**: Matthew Zeiler and Rob Fergus
- **Year**: 2013
- **Goal**: Understand and improve upon AlexNet through visualization
- **Key Contribution**: Introduced deconvolution-based visualization techniques

### Architecture Changes from AlexNet

| Layer | AlexNet | ZF-Net |
|-------|---------|--------|
| Conv1 | 11×11, stride 4 | 7×7, stride 2 |
| Conv2 | 5×5 | 5×5 |
| Conv3-5 | 3×3 | 3×3 |
| Total Filters | Similar | Adjusted based on visualization insights |

### Why These Changes?
- **Smaller first filters**: Better capture of fine-grained details
- **Reduced stride**: Maintain more spatial information
- **Analysis**: Used visualization to identify bottlenecks

### Visualization Contribution
- **Deconvolution Visualization**: Show what each filter learns
- **Revealed**: Feature hierarchy and important patterns
- **Impact**: Influenced VGGNet and other subsequent architectures

---

## VGGNet

### Overview
- **Developers**: Visual Geometry Group, University of Oxford
- **Year**: 2014
- **Variants**: VGG-16, VGG-19, VGG-11
- **Achievement**: 2nd place ILSVRC 2014
- **Key Insight**: Simplicity and depth matter more than complex architectures

### Philosophy
> "The representation depth is beneficial for classification accuracy" - VGGNet Paper

**Main Principle**: Use **very small 3×3 filters** with **stride 1** and **deep networks**

### VGGNet-16 Detailed Architecture (PYQ)

#### **Input Layer**
- **Size**: 224×224×3 (RGB images)
- **Preprocessing**: Subtract ImageNet mean

#### **Block 1 (Layers 1-2)**
```
Input (224×224×3)
  ↓
Conv2D(64, 3×3, padding='same') + ReLU
  ↓
Conv2D(64, 3×3, padding='same') + ReLU → Output: 224×224×64
  ↓
MaxPooling2D(2×2, stride 2) → Output: 112×112×64
```

#### **Block 2 (Layers 3-4)**
```
Input (112×112×64)
  ↓
Conv2D(128, 3×3, padding='same') + ReLU
  ↓
Conv2D(128, 3×3, padding='same') + ReLU → Output: 112×112×128
  ↓
MaxPooling2D(2×2, stride 2) → Output: 56×56×128
```

#### **Block 3 (Layers 5-7)**
```
Input (56×56×128)
  ↓
Conv2D(256, 3×3, padding='same') + ReLU
  ↓
Conv2D(256, 3×3, padding='same') + ReLU
  ↓
Conv2D(256, 3×3, padding='same') + ReLU → Output: 56×56×256
  ↓
MaxPooling2D(2×2, stride 2) → Output: 28×28×256
```

#### **Block 4 (Layers 8-10)**
```
Input (28×28×256)
  ↓
Conv2D(512, 3×3, padding='same') + ReLU
  ↓
Conv2D(512, 3×3, padding='same') + ReLU
  ↓
Conv2D(512, 3×3, padding='same') + ReLU → Output: 28×28×512
  ↓
MaxPooling2D(2×2, stride 2) → Output: 14×14×512
```

#### **Block 5 (Layers 11-13)**
```
Input (14×14×512)
  ↓
Conv2D(512, 3×3, padding='same') + ReLU
  ↓
Conv2D(512, 3×3, padding='same') + ReLU
  ↓
Conv2D(512, 3×3, padding='same') + ReLU → Output: 14×14×512
  ↓
MaxPooling2D(2×2, stride 2) → Output: 7×7×512
```

#### **Fully Connected Layers (Layers 14-16)**
```
Flatten: 7×7×512 = 25,088
  ↓
Dense(4096) + ReLU (Layer 14)
  ↓
Dense(4096) + ReLU (Layer 15)
  ↓
Dense(1000) + Softmax (Layer 16 - Output)
```

### VGGNet Layer Summary Table

| Layer # | Type | Filters | Filter Size | Stride | Padding | Output Size |
|---------|------|---------|------------|--------|---------|------------|
| 1-2 | Conv | 64 | 3×3 | 1 | same | 224×224×64 |
| - | MaxPool | - | 2×2 | 2 | - | 112×112×64 |
| 3-4 | Conv | 128 | 3×3 | 1 | same | 112×112×128 |
| - | MaxPool | - | 2×2 | 2 | - | 56×56×128 |
| 5-7 | Conv | 256 | 3×3 | 1 | same | 56×56×256 |
| - | MaxPool | - | 2×2 | 2 | - | 28×28×256 |
| 8-10 | Conv | 512 | 3×3 | 1 | same | 28×28×512 |
| - | MaxPool | - | 2×2 | 2 | - | 14×14×512 |
| 11-13 | Conv | 512 | 3×3 | 1 | same | 14×14×512 |
| - | MaxPool | - | 2×2 | 2 | - | 7×7×512 |
| 14 | Dense | 4096 | - | - | - | 4096 |
| 15 | Dense | 4096 | - | - | - | 4096 |
| 16 | Dense | 1000 | - | - | - | 1000 |

### Why Small 3×3 Filters?

#### **Receptive Field Equivalence**:
- **Two 3×3 filters** = **One 5×5 filter** (receptive field)
- **Three 3×3 filters** = **One 7×7 filter** (receptive field)

#### **Advantages**:
```
Two 3×3 filters:
  - Parameters: 2 × (3×3) = 18
  - Non-linearities: 2 (ReLU applied twice)

One 5×5 filter:
  - Parameters: 1 × (5×5) = 25
  - Non-linearities: 1 (ReLU applied once)

Result: Small filters with MORE non-linearities = Better learning capacity!
```

### Key Features of VGGNet

1. **Depth**: Up to 19 layers (VGG-19)
2. **Simplicity**: Uniform 3×3 filters throughout
3. **Regularization**: L2 weight decay, early stopping
4. **Computational Cost**: High (memory and time intensive)
5. **Number of Parameters**: 138M (VGG-16), 144M (VGG-19)

### VGGNet Code Implementation (Keras/TensorFlow)

```python
def VGG16():
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', 
                     activation='relu', input_shape=(224, 224, 3)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 4
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 5
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1000, activation='softmax')
    ])
    
    return model

model = VGG16()
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

### VGGNet Comparison Table

| Variant | Layers | Conv Layers | FC Layers | Parameters | ImageNet Top-5 |
|---------|--------|-------------|-----------|------------|-----------------|
| VGG-11 | 11 | 8 | 3 | 132M | 89.8% |
| VGG-13 | 13 | 10 | 3 | 133M | 90.4% |
| VGG-16 | 16 | 13 | 3 | 138M | 91.3% |
| VGG-19 | 19 | 16 | 3 | 144M | 91.5% |

---

## GoogLeNet (Inception Network)

### Overview
- **Developers**: Google team
- **Year**: 2014
- **Achievement**: Won ILSVRC 2014
- **Key Innovation**: Inception modules (multi-scale feature extraction)

### Problem GoogLeNet Solved
- **Increasing depth** → Computational complexity increases exponentially
- **Large filter sizes** → High computational cost
- Solution: **Inception Module** - parallel multi-scale processing

### Inception Module Architecture

The Inception module uses **1×1, 3×3, and 5×5 convolutions in parallel**:

```
Input
  ├─→ Conv2D(1×1) ────────────────┐
  ├─→ Conv2D(1×1) → Conv2D(3×3) ──┤
  ├─→ Conv2D(1×1) → Conv2D(5×5) ──┼─→ Concatenate → Output
  └─→ MaxPool(3×3) → Conv2D(1×1) ─┘
```

#### **Purpose of 1×1 Convolution**:
- **Dimensionality reduction**: Reduce channels before expensive 3×3, 5×5 operations
- **Computation savings**: Reduce parameters and FLOPs
- **Non-linearity**: Add non-linear transformations

### GoogLeNet Architecture (22 layers)

#### **Key Components**:
1. **Stem (Initial Convolutions)**: 224×224×3 → feature maps
2. **Inception Modules (3)**: 3 groups of inception blocks
3. **Global Average Pooling**: Replace FC layers (reduces parameters)
4. **Output Layer**: 1000 classes with Softmax

#### **Auxiliary Classifiers**:
- Two auxiliary classifiers added mid-network
- **Purpose**: Provide gradient signals deeper in network during backprop
- **Removed**: During inference

### Advantages of GoogLeNet
1. **Efficiency**: 12× fewer parameters than AlexNet
2. **Speed**: 4× reduction in computation
3. **Depth**: 22 layers without vanishing gradient
4. **Modularity**: Inception blocks are reusable

---

## ResNet (Residual Networks)

### Overview
- **Developers**: Microsoft Research (He et al.)
- **Year**: 2015
- **Variants**: ResNet-50, ResNet-101, ResNet-152
- **Achievement**: Won ILSVRC 2015
- **Key Innovation**: Residual connections (skip connections)

### The Degradation Problem
- **Problem**: Increasing network depth → Accuracy decreases on both train and test sets
- **Not due to overfitting**: Training error also increases
- **Reason**: Vanishing gradient problem; hard to optimize very deep networks

### Residual Connection (Skip Connection)

#### **Traditional Path**:
```
x → [Conv2D] → [Conv2D] → y
```

#### **Residual Path**:
```
x ──────────────────┐
  │                 │
  └─→ [Conv2D] ──→ [Conv2D] ──┐
                                ↓
                           y = F(x) + x
```

#### **Mathematical Formulation**:
$$H(x) = F(x) + x$$

Where:
- $H(x)$ = desired output
- $F(x)$ = learned residual mapping
- $x$ = input (skip connection)

#### **Why This Works**:
- **Backpropagation**: Gradient can flow directly through skip connection
- **Optimization**: Easier to learn residual mapping $F(x)$ than identity mapping
- **Vanishing Gradient**: Skip connection provides direct gradient path

### ResNet Architecture

#### **Residual Block Types**:

**1. Basic Block** (used in ResNet-34):
```
Input (H×W×C)
  ↓
Conv2D(3×3) + BatchNorm + ReLU
  ↓
Conv2D(3×3) + BatchNorm
  ↓
Add input (skip connection)
  ↓
ReLU
  ↓
Output (H×W×C)
```

**2. Bottleneck Block** (used in ResNet-50+):
```
Input (H×W×C)
  ↓
Conv2D(1×1, C/4) + BatchNorm + ReLU  [Dimension reduction]
  ↓
Conv2D(3×3, C/4) + BatchNorm + ReLU  [Main computation]
  ↓
Conv2D(1×1, C) + BatchNorm            [Dimension expansion]
  ↓
Add input (skip connection)
  ↓
ReLU
  ↓
Output (H×W×C)
```

### ResNet Architecture Comparison

| Model | Layers | Parameters | ImageNet Top-5 |
|-------|--------|------------|-----------------|
| ResNet-18 | 18 | 11.7M | 89.3% |
| ResNet-34 | 34 | 21.8M | 90.3% |
| ResNet-50 | 50 | 25.5M | 91.2% |
| ResNet-101 | 101 | 44.5M | 92.0% |
| ResNet-152 | 152 | 60.2M | 92.4% |

### Key Features
1. **Depth**: Up to 152 layers trained successfully
2. **Efficiency**: Fewer parameters than VGGNet despite more layers
3. **Batch Normalization**: Used throughout for training stability
4. **Skip Connections**: Enable very deep networks

### ResNet Code Implementation (Keras/TensorFlow)

```python
def residual_block(x, filters, kernel_size=3, stride=1, 
                   use_bottleneck=False):
    if use_bottleneck:
        # Bottleneck block
        y = layers.Conv2D(filters//4, (1, 1), padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        y = layers.Conv2D(filters//4, (kernel_size, kernel_size), 
                         stride=stride, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        y = layers.Conv2D(filters, (1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)
    else:
        # Basic block
        y = layers.Conv2D(filters, (kernel_size, kernel_size), 
                         stride=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        y = layers.Conv2D(filters, (kernel_size, kernel_size), 
                         padding='same')(y)
        y = layers.BatchNormalization()(y)
    
    # Skip connection
    if stride != 1:
        x = layers.Conv2D(filters, (1, 1), stride=stride, 
                         padding='same')(x)
        x = layers.BatchNormalization()(x)
    
    y = layers.Add()([y, x])
    y = layers.Activation('relu')(y)
    
    return y

def ResNet50(input_shape=(224, 224, 3), num_classes=1000):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 256, stride=1, use_bottleneck=True)
    x = residual_block(x, 256, stride=1, use_bottleneck=True)
    x = residual_block(x, 256, stride=1, use_bottleneck=True)
    
    x = residual_block(x, 512, stride=2, use_bottleneck=True)
    # ... (more blocks)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

---

## CNN Visualization Techniques

### Why Visualize Neural Networks?
- **Interpretability**: Understand what networks learn
- **Debugging**: Identify problems in training
- **Trust**: Ensure models make decisions for right reasons
- **Research**: Discover new insights about deep learning

### 1. Feature Visualization (Activation Visualization)

#### **What It Shows**:
- What features each filter learns to detect
- How different layers represent information

#### **Method**:
```
Input Image
  ↓
Pass through network
  ↓
Visualize activation maps at each layer
  ↓
Early layers: Simple patterns (edges, colors)
  ↓
Middle layers: Textures, shapes
  ↓
Deep layers: Complex objects
```

#### **Interpretation**:
- **Layer 1**: Detects edges, corners, blobs
- **Layer 2**: Combines edges → textures, simple shapes
- **Layer 3-4**: Complex patterns, object parts
- **Layer 5+**: High-level object concepts

---

### 2. Guided Backpropagation

#### **What It Shows**:
- Which input pixels contribute most to specific class prediction

#### **How It Works**:

```
1. Forward Pass: Compute network output for input image
2. Select Class: Choose target class to visualize
3. Backward Pass: Backpropagate error with modification:
   - Only propagate positive gradients
   - Ignore negative gradients
4. Gradient Map: Visualize gradients at input layer
```

#### **Mathematical Formulation**:
- **Standard Backprop Gradient**:
  $$g = \frac{\partial L}{\partial x}$$

- **Guided Backprop Modification**:
  $$g_{guided} = g \cdot \mathbb{1}_{g > 0} \cdot \mathbb{1}_{a > 0}$$
  
  Where:
  - $\mathbb{1}_{g > 0}$ : Only positive gradients
  - $\mathbb{1}_{a > 0}$ : Only positive activations

#### **Advantages**:
- Produces cleaner, more interpretable visualizations
- Shows relevant input regions for decision
- Works across different architectures

#### **Code Example** (Conceptual):
```python
def guided_backprop(model, input_image, class_index):
    # Forward pass
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image)
        loss = predictions[:, class_index]
    
    # Backward pass with modification
    gradients = tape.gradient(loss, input_image)
    
    # Apply guidance: only positive gradients
    guided_gradients = tf.cast(gradients > 0, tf.float32) * gradients
    
    # Visualize
    visualization = tf.reduce_sum(tf.abs(guided_gradients), axis=-1)
    return visualization
```

---

### 3. Deep Dream

#### **What It Is**:
- Artistic visualization technique
- Makes hidden features in network explicit
- Creates surreal, dreamlike images

#### **How It Works**:

```
1. Start with: Input image or random noise
2. Modify Goal: Instead of minimizing loss,
                MAXIMIZE activation of chosen layer
3. Iterative Process:
   - Compute activation for chosen layer
   - Calculate gradients w.r.t. input image
   - Update input to increase activation
   - Repeat multiple times
```

#### **Mathematical Process**:
$$x_{t+1} = x_t + \alpha \cdot \frac{\partial a_l}{\partial x_t}$$

Where:
- $x_t$ : Input image at iteration t
- $\alpha$ : Learning rate (step size)
- $a_l$ : Activation of layer l
- Goal: Maximize layer activation (amplify learned features)

#### **Parameters**:
- **Layer Selection**: Deeper layers → more complex patterns
- **Step Size**: Large → fast changes, small → subtle changes
- **Iterations**: More → stronger effect

#### **Code Example** (Conceptual):
```python
def deep_dream(model, input_image, layer_name, num_iterations=100, step_size=0.01):
    # Get layer to visualize
    layer = model.get_layer(layer_name)
    dream_model = keras.Model(inputs=model.input, 
                              outputs=layer.output)
    
    # Convert to variable
    x = tf.Variable(input_image, dtype=tf.float32)
    
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            activations = dream_model(x)
            loss = tf.reduce_mean(activations)  # Maximize activation
        
        gradients = tape.gradient(loss, x)
        x.assign_add(step_size * gradients)  # Update image
    
    return x.numpy()
```

#### **Results**:
- Reveals what network considers important features
- Creates artistic, often pareidolic images
- Shows learned hierarchical representations

---

### 4. Deep Art (Neural Style Transfer)

#### **What It Is**:
- Technique to apply artistic style from one image to another
- Uses deep networks to separate content and style

#### **Key Concept**:
- **Content**: What the image depicts (objects, structure)
- **Style**: How it's painted (color, texture, brushstrokes)

#### **How It Works**:

```
Input:
  - Content Image: Your photograph
  - Style Image: Famous artwork
  
Process:
  1. Extract content features from content image (deep layers)
  2. Extract style features from style image (Gram matrix)
  3. Generate new image that matches:
     - Content features (preserve object structure)
     - Style features (apply artistic style)
  
Output:
  - Image with content of photo + style of artwork
```

#### **Feature Extraction**:

**Content Features**:
- Use activation maps from mid-to-deep layers
- Preserves spatial structure and objects

**Style Features**:
- Use Gram Matrix of activation maps
- Captures texture and color relationships
- Gram Matrix: $G_{ij} = \sum_k a_{ik} \cdot a_{jk}$

#### **Loss Function**:
$$L_{total} = \alpha \cdot L_{content} + \beta \cdot L_{style}$$

Where:
- $L_{content}$ : MSE between content features
- $L_{style}$ : Difference in Gram matrices
- $\alpha, \beta$ : Weighting parameters

#### **Code Example** (Conceptual):
```python
def style_transfer(content_image, style_image, num_iterations=1000):
    # Pretrained network (e.g., VGG19)
    vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
    
    # Extract content and style features
    content_features = vgg(content_image)
    style_features = vgg(style_image)
    
    # Generate image (initialize with content)
    generated = tf.Variable(content_image, dtype=tf.float32)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.02)
    
    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            gen_features = vgg(generated)
            
            # Content loss
            content_loss = tf.reduce_mean(
                tf.square(gen_features - content_features)
            )
            
            # Style loss (Gram matrix)
            style_loss = gram_matrix_loss(gen_features, style_features)
            
            total_loss = content_loss + 0.01 * style_loss
        
        gradients = tape.gradient(total_loss, generated)
        optimizer.apply_gradients([(gradients, generated)])
    
    return generated.numpy()
```

---

### 5. Fooling Convolutional Neural Networks (Adversarial Examples)

#### **What It Is**:
- Adversarial examples that fool neural networks
- Demonstrate robustness vulnerabilities of deep networks
- Small perturbations → completely wrong predictions

#### **Example**:
```
Original Image: Panda (99% confidence)
  + Small Noise (imperceptible to humans)
  ↓
Adversarial Image: Looks like Panda to humans
                   but CNN predicts Gibbon (99% confidence)
```

#### **Why This Happens**:
- Networks optimize for human-understandable features
- Adversarial perturbations exploit decision boundaries
- Tiny changes in pixel values can shift predictions

#### **Creation Methods**:

**1. Fast Gradient Sign Method (FGSM)**:
```
1. Compute gradient of loss w.r.t. input: ∇_x L
2. Create perturbation: δ = ε · sign(∇_x L)
3. Adversarial example: x' = x + δ
```

**Code**:
```python
def fgsm_attack(model, input_image, true_label, epsilon=0.3):
    input_image = tf.Variable(input_image, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        predictions = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            [true_label], predictions
        )
    
    gradients = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradients)
    
    adversarial = input_image + epsilon * signed_grad
    return tf.clip_by_value(adversarial, 0, 1)
```

**2. Iterative FGSM (I-FGSM)**:
- Apply FGSM multiple times with small steps
- More powerful than single-step FGSM
- Creates stronger adversarial examples

**3. Carlini-Wagner Attack**:
- Optimization-based approach
- Find minimal perturbation that fools network
- More effective than FGSM

#### **Implications**:
1. **Security**: Models vulnerable to adversarial attacks
2. **Robustness**: Need adversarial training
3. **Generalization**: Models don't generalize like humans
4. **Defense**: Adversarial training, input normalization

#### **Defense Strategies**:
- **Adversarial Training**: Train on adversarial examples
- **Input Preprocessing**: Normalize, denoise inputs
- **Model Ensemble**: Use multiple models
- **Certified Defenses**: Provable robustness bounds

---

## ImageNet Dataset

### Overview
- **Total Images**: Over 14 million
- **Categories**: 21,000+
- **Challenge**: ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

### Categories Include:

#### **Animals**:
- Mammals (dogs, cats, lions, elephants)
- Birds (eagles, parrots, owls)
- Reptiles & Amphibians (snakes, frogs, lizards)
- Aquatic Animals (fish, dolphins, whales)

#### **Objects & Artifacts**:
- Everyday items (chairs, tables, bags, shoes)
- Electronics (laptops, cameras, mobile phones)
- Vehicles (cars, bicycles, airplanes, boats)
- Musical instruments (guitars, drums, pianos)

#### **Scenes & Environments**:
- Natural landscapes (mountains, beaches, forests)
- Urban settings (streets, bridges, buildings)

### ILSVRC Challenge
- **Annual competition** (2010-2017)
- **Tasks**: Image classification, localization, detection
- **Driving force** for CNN innovation (AlexNet, VGGNet, ResNet, etc.)

---

## Summary & Key Takeaways

### Model Evolution Timeline
```
1998: LeNet-5 (5 layers, 60K params)
  ↓
2012: AlexNet (8 layers, 60M params, ReLU, GPU training)
  ↓
2013: ZF-Net (Visualization insights)
  ↓
2014: VGGNet (19 layers, 3×3 filters, 144M params)
      GoogLeNet (Inception modules, 22 layers, 4M params)
  ↓
2015: ResNet (152 layers, skip connections, 60M params)
```

### Key Innovations
1. **LeNet**: Introduced CNNs for practical applications
2. **AlexNet**: ReLU, dropout, GPU training → revolutionized deep learning
3. **ZF-Net**: Visualization techniques for understanding networks
4. **VGGNet**: Showed that depth matters; simple is effective
5. **GoogLeNet**: Inception modules for efficiency
6. **ResNet**: Skip connections enabled very deep networks

### Visualization Importance
- **Feature Maps**: Understand learned representations
- **Guided Backprop**: Identify important input regions
- **Deep Dream**: Reveal network's learned features
- **Neural Style Transfer**: Create artistic images
- **Adversarial Examples**: Test and improve robustness

### For Exam Preparation
**Must Know**:
- ✅ LeNet architecture and feature learning mechanism
- ✅ AlexNet innovations and use cases
- ✅ VGGNet layer-by-layer architecture
- ✅ ResNet skip connections and why they work
- ✅ All visualization techniques and their applications

**Important Concepts**:
- Feature hierarchy: simple → complex features
- Gradient flow through deep networks
- Parameter efficiency: more layers ≠ more parameters
- Transfer learning: pre-training and fine-tuning

---

## Important Formulas & Equations

### Convolutional Layer Output Size
$$H_{out} = \frac{H_{in} - F + 2P}{S} + 1$$
$$W_{out} = \frac{W_{in} - F + 2P}{S} + 1$$

Where:
- $H_{in}, W_{in}$ : Input height, width
- $F$ : Filter size
- $P$ : Padding
- $S$ : Stride

### Receptive Field
$$RF_n = RF_{n-1} + (F_n - 1) \times \prod_{i=1}^{n-1} S_i$$

### Gram Matrix
$$G_{ij}^l = \sum_k a_{ik}^l \cdot a_{jk}^l$$

### ResNet Residual Connection
$$H(x) = F(x) + x$$

### Guided Backprop Gradient
$$g_{guided} = g \cdot \mathbb{1}_{g > 0} \cdot \mathbb{1}_{a > 0}$$

---

**Document prepared for Unit 3 Mid Examination**
**All architectures, code examples, and visualizations as per curriculum**
