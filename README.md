# Learning-Computer-Vision-with-DSal
by Ahmad Salehi

# 1. Introduction
Computer Vision (CV) is a field of AI that enables machines to interpret and process visual data (images/videos) to extract useful information. It involves tasks like object detection, image classification, segmentation, and recognition.
CV enables computers to interpret and analyze visual data (images/videos) like humans. It involves feature extraction, pattern recognition, and decision-making using machine learning and deep learning.


| Field | Definition | Common Uses | Wavelengths & Sensors |
|-------|------------|-------------|-----------------------|
| Image Processing | Enhancing or modifying images without understanding their content | Noise removal, filtering, edge detection, compression | Optical cameras (RGB, InfraRed), scanners|
| Computer Vision | Teaching machines to interpret images like humans do |Object recognition, facial recognition, autonomous driving | RGB cameras, depth sensors (LiDAR), Infrared cameras |
| Machine Vision | Industrial use of vision systems for inspection and automation | Quality control, robotics, barcode scanning | InfraRed (thermal), X-ray, multispectral cameras|

Table 1_1: Image processing vs. computer vision vs. machine vision


| Wavelength Range | Type | Sensor Used | Applications |
|-------|------------|-------------|-----------------------|
| 0.01-10 nm | X-ray Vision | X-ray detectors | Medical purposes |
| 400-730 nm | Visible Light (RGB) | CMOS, CCD | Image classification, object detection |
| 730-1400 nm | Near-InfraRed (NIR) |NIR Cameras | Eye tracking, night vision |
| 1.4-3 um | Short-wave InfraRed (SWIR) | SWIR Cameras | Eye tracking, night vision |
| 3-14 um | Thermal InfraRed (TIR) | Uncooled microbolometers | Heat detection, surveillance, medical imaging |
| 1 mm -  1 m | Radar | Radar sensors | Automotive collision detection |

Table 1_1: Wavelengths and optical sensors used

# 1.1. Key components of computer vision
Input: Information Retrieval
Accessing and extracting relevant data from large datasets based on content.
How It Relates to CV:
  - Image Search & Retrieval: Finding similar images in a database.
  - Feature Extraction: Identifying patterns (edges, colors, textures).
  - Reverse Image Search: Matching a given image to known references.
    Example: Google Reverse Image Search uses CV to find visually similar images online.
    
Output: Machine Learning
Using algorithms to learn from image data and make predictions.
How It Relates to CV:
  - Supervised Learning: Training a model using labeled datasets (e.g., ImageNet for classification).
  - Unsupervised Learning: Clustering images without predefined labels (e.g., anomaly detection).
  - Deep Learning (CNNs, Transformers): Models that analyze pixel patterns to recognize objects.
    Example: Self-driving cars use CNNs (Convolutional Neural Networks) to identify road signs, pedestrians, and obstacles.
    Results: Mathematics
    Definition: Mathematical principles that form the backbone of Computer Vision.
    
# 1.2. Key areas in computer vision
  - Linear Algebra: Matrix operations for image transformations (rotation, scaling).
  - Calculus: Optimization (gradient descent for training deep learning models).
  - Probability & Statistics: Bayesian filtering, noise reduction, and uncertainty modeling.
  - Geometry: 3D reconstruction, depth estimation, and camera calibration. Example: Epipolar Geometry is used in stereo vision to estimate 3D depth from two 2D images.

# 1.3. How these components work together
  - Input (Information Retrieval): Collects and preprocesses visual data.
  - Output (Machine Learning): Analyzes and learns patterns from the data.
  - Results (Mathematics): Ensures accurate transformations, optimizations, and model reliability.

# 1.4. Core fields of computer vision
  - Computer Intelligence → The broad AI-driven approach to decision-making and pattern recognition.
  - Artificial Intelligence (AI) → CV is a subset of AI that processes visual data.
  - Cognitive Vision → Goes beyond recognition, incorporating context and reasoning.
  - Machine Learning (ML) → Deep learning architectures (CNNs, Transformers, GANs) drive modern CV.
  - Mathematical Foundations:
    Statistics → Bayesian networks, probabilistic models. | Geometry → 3D reconstruction, camera calibration. | Optimization → Gradient descent, convex optimization.

# 1.5. Vision-specific applications
  - Robotic Vision → SLAM (Simultaneous Localization and Mapping), obstacle detection.
  - Control Robotics → Vision-based control systems in autonomous vehicles.
  - Computer Vision → High-level interpretation (e.g., face recognition, medical imaging).
  - Image Processing → Low-level transformations (e.g., Fourier Transform, edge detection).
  - Neurobiology & Biological Vision → Bio-inspired AI (e.g., CNNs inspired by the visual cortex).

# 1.6. Technical foundations
  - Signal Processing
  - Multi-variable Signal Processing → Used for hyperspectral imaging, video analysis.
  - Non-linear Signal Processing → Advanced feature extraction techniques.
  - Physics & Imaging → Optics, LiDAR, radiance models.
  - Smart Cameras → On-device real-time processing (e.g., FPGA-based vision).

# 1.7. How these areas connect
  - Physics & Imaging → Models how light interacts with surfaces (radiance, reflections).
  - Signal Processing → Transforms raw sensor data (Fourier transforms, wavelets).
  - Image Processing → Enhances and segments images for analysis.
  - Computer Vision & AI → Recognizes objects, scenes, patterns.
  - Robotics & Smart Applications → Uses vision for automation and real-world interaction.

# 1.8. Deep learning for computer vision
Deep learning has revolutionized computer vision by replacing handcrafted features with end-to-end learning models. Key Architectures and methods of deep learning for CV are:
  - Convolutional Neural Networks (CNNs) → Feature extraction from images. Examples: ResNet, VGG, EfficientNet.
  - Transformers in Vision (ViTs, Swin-Transformer) → Attention-based models replacing CNNs.
  - Generative Adversarial Networks (GANs) → Image generation, super-resolution.
  - Reinforcement Learning (RL) for Vision → Used in robotics for real-world decision-making.
In addition, applications of deep learning for CV are:
  - object detection & segmentation (YOLO, Mask R-CNN).
  - Medical imaging (AI-based tumor detection).
  - Autonomous driving (Tesla uses deep vision networks).
  - 
# 1.9. Hardwave aspects of computer vision: FPGA & Edge AI
To make computer vision faster and more efficient, we use specialized hardware.
FPGA (Field-Programmable Gate Arrays): FPGA provides low-latency, high-performance computation for vision applications.
  - Used in: Real-time object tracking, autonomous drones, embedded vision.
  - Examples: Xilinx Zynq, Intel Arria.
Edge AI: Edge AI moves AI processing to local devices instead of cloud computing.
  - Benefits: Faster inference, lower power consumption.
  - Examples: NVIDIA Jetson (for robotics & autonomous systems), Google Coral TPU (for AI at the edge), and Intel Movidius (for smart cameras).
