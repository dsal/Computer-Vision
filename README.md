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
**Input**: Information Retrieval
Accessing and extracting relevant data from large datasets based on content.
How It Relates to CV:
  - **Image Search & Retrieval**: Finding similar images in a database.
  - **Feature Extraction**: Identifying patterns (edges, colors, textures).
  - **Reverse Image Search**: Matching a given image to known references.
    Example: Google Reverse Image Search uses CV to find visually similar images online.
    
**Output**: Machine Learning
Using algorithms to learn from image data and make predictions.
How It Relates to CV:
  - **Supervised Learning**: Training a model using labeled datasets (e.g., ImageNet for classification).
  - **Unsupervised Learning**: Clustering images without predefined labels (e.g., anomaly detection).
  - **Deep Learning (CNNs, Transformers)**: Models that analyze pixel patterns to recognize objects.
    Example: Self-driving cars use CNNs (Convolutional Neural Networks) to identify road signs, pedestrians, and obstacles.
    Results: Mathematics
    Definition: Mathematical principles that form the backbone of Computer Vision.
    
# 1.2. Key areas in computer vision
CV relies on foundational concepts from several core areas of mathematics and computer science. These key areas provide the theoretical and practical tools needed to process, analyze, and extract meaningful information from images and videos.
  - **Linear Algebra**: for manipulating image data through matrix operations, such as transformations (rotation, scaling) and convolutional filters.
  - **Calculus**: (particularly optimization techniques like gradient descent) for training deep learning models that improve vision-based tasks.
  - **Probability & Statistics**: for handling uncertainty, noise reduction, and probabilistic models like Bayesian filtering.
  - **Geometry**: (especially projective geometry) for tasks such as 3D reconstruction, depth estimation, and camera calibration. For example, epipolar geometry is crucial in stereo vision systems to derive 3D structure from multiple 2D images.
Understanding these areas is fundamental to developing robust computer vision systems, from simple image filters to advanced applications like autonomous driving and medical imaging.

# 1.3. How these components work together
CV systems function as a cohesive pipeline, integrating multiple stages—input processing, machine learning analysis, and mathematical validation, to transform raw visual data into actionable insights. Each component plays a critical role in ensuring accuracy, efficiency, and reliability.
  - **Input (Information Retrieval & Preprocessing)**: Collects and preprocesses visual data. Raw images/videos are acquired (e.g., from cameras or sensors) and refined for analysis. In this step, key tasks include noise reduction (e.g., Gaussian filtering), normalization (rescaling pixel values), and feature extraction (e.g., edges, textures). For instance, a self-driving car’s cameras capture street scenes, which are then cropped and adjusted for lighting.
  - **Output (Machine Learning & Pattern Recognition)**: Analyzes and learns patterns from the data. The purpose of this step is modeling (e.g., CNNs, transformers), detecting and interpreting patterns. In this step, key tasks include In this step, key tasks include, image classification (ResNet, ViT), and semantic segmentation (U-Net). For instance, a medical AI system identifies tumors in X-rays by learning from labeled datasets.
  - **Results (Mathematics)**: Ensures accurate transformations, optimizations, and model reliability. The purpose is ensuring reliablity and efficiency of output. In this step, key tasks include geometric verification (e.g., epipolar constraints in stereo vision), statistical confidence scores (e.g., Bayesian inference), optimization (e.g., backpropagation tuning model weights). For instance, a drone uses 3D triangulation (geometry) to validate its depth-map accuracy.

# 1.4. Core fields of computer vision
CV merges techniques from artificial intelligence, cognitive science, machine learning, and mathematics to enable machines to see and interpret visual data. These core fields work synergistically to solve complex problems—from object detection to autonomous navigation. Below is a breakdown of their roles and interconnections:
  - **Computer Intelligence** → The broad AI-driven approach to decision-making and pattern recognition.
  - **Artificial Intelligence (AI)** → CV is a subset of AI that processes visual data.
  - **Cognitive Vision** → Goes beyond recognition, incorporating context and reasoning.
  - **Machine Learning (ML)** → Deep learning architectures (CNNs, Transformers, GANs) drive modern CV.
  - **Mathematical Foundations**:
    Statistics → Bayesian networks, probabilistic models. | Geometry → 3D reconstruction, camera calibration. | Optimization → Gradient descent, convex optimization.

# 1.5. Vision-specific applications
CV powers a diverse range of applications, each leveraging unique techniques to solve real-world problems. From enabling robots to navigate autonomously to revolutionizing medical diagnostics, these applications highlight the transformative potential of CV. Below is a structured overview of key vision-specific domains and their significance:
  - **Robotic Vision** → SLAM (Simultaneous Localization and Mapping: Allows robots to map unknown spaces while tracking their location.), obstacle detection (Uses depth sensors (LiDAR) or stereo cameras to avoid collisions.)
  - **Control Robotics** → Vision-based control systems in autonomous vehicles. It integrates vision into decision-making for autonomous systems. Applications include autonomous Vehicles (Cameras + LiDAR detect lanes, pedestrians, and traffic signs.), and industrial Robots (Vision-guided arms assemble products or sort items in warehouses.)
  - **Computer Vision** → High-level interpretation (e.g., face recognition, medical imaging); Extracting semantic meaning from visual data. The key tasks include face recognition (Biometric security), and medical imaging (detecting tumors in MRI scans or diabetic retinopathy in retinal images)
  - **Image Processing** → Low-level transformations (e.g., Fourier Transform, edge detection); Enhancing or manipulating raw pixel data. The key techniques are fourier transform (Decomposes images into frequencies used in compression, like JPEG.), and edge detection (identifies object boundaries.)
  - **Neurobiology & Biological Vision** → Bio-inspired AI (e.g., CNNs inspired by the visual cortex); drawing inspiration from human/animal vision to improve AI such as CNN Inspiration (Mimics the visual cortex’s hierarchical processing), and Bio-Inspired Sensors (Event cameras that replicate the retina’s response to light changes)

# 1.6. Technical foundations
The field of CV is built on rigorous technical foundations that enable machines to process, analyze, and interpret visual data. These foundations draw from signal processing, physics, and hardware innovations to address challenges like noise reduction, real-time processing, and 3D reconstruction. Below is a detailed breakdown of these core technical pillars:
  - **Signal Processing**: The backbone of low-level image analysis. Key techniques include Fourier Transform (Converts images to frequency domains for compression,JPEG , or noise removal), and filtering (Gaussian/median filters smooth images or enhance edges). For instance, MRI scans use Fourier-based reconstruction to convert raw signals into images.
  - **Multi-variable Signal Processing** → Handles complex, high-dimensional data. The application include Hyperspectral imaging (Captures hundreds of spectral bands: used in agriculture to monitor crop health), and video analysis (Temporal signal processing for motion tracking, e.g., sports analytics).
  - **Non-linear Signal Processing** → Advanced feature extraction techniques; it extracts intricate patterns where linear methods fail. The techniques are wavelet transforms (multi-resolution analysis for texture classification), and morphological operations (shape-based filtering, e.g., erosion/dilation in medical imaging). For example detecting tumor boundaries in ultrasound images using non-linear edge enhancement.
  - **Physics & Imaging** → Models how light interacts with scenes and sensors. Key areas include optics (Lens design, aberrations, and focus models), LiDAR (Time-of-flight principles for 3D depth mapping), and radiance models (Simulates light transport, e.g., for virtual reality rendering). For example, autonomous vehicles use LiDAR physics to correct for weather-related noise.
  - **Smart Cameras** → embeds processing power directly into cameras. The techniques are FPGA/ASIC Chips (Enable real-time edge detection or object tracking without a PC) and event cameras (Mimic the human retina with microsecond latency used in high-speed robotics). For example, surveillance cameras with on-device facial recognition.

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
