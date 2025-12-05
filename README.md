# Computer Vision: Corner Detection & Image Matching
**A comprehensive Computer Vision tool built with Python, NumPy, and PyQt5.**

## Project Overview
This application provides a dual-panel GUI to visualize and experiment with fundamental feature detection algorithms. Unlike standard implementations, the core algorithms here are **built from scratch** using NumPy to demonstrate a deep understanding of the underlying mathematics. OpenCV is used strictly for image loading and saving.

## Key Features
* **Custom Algorithms:** Pure Python implementations of **Harris Corner Detection**, **Shi-Tomasi**, and **SIFT** (Scale-Invariant Feature Transform).
* **Interactive GUI:** Built with **PyQt5**, featuring real-time sliders to adjust thresholds, K-parameters, and corner counts.
* **Deep Visualization:** View Difference of Gaussian (DoG) pyramids, gradient orientations, and keypoint characteristics using **Matplotlib**.
* **Image Matching:** Compare two images using SIFT features to detect corresponding keypoints, robust against scaling and rotation changes.

## Screenshots
**1. Harris Corner Detection**

<img width="1170" height="624" alt="image" src="https://github.com/user-attachments/assets/2598bc33-3cd5-4dce-8f96-0ec4ae17f3c4" />


**2. Image Matching Visualization**

<img width="1167" height="621" alt="image" src="https://github.com/user-attachments/assets/1213a9cf-415e-4d9b-ba56-d65b9cbab834" />

## How to Run
1.  **Install Dependencies:**
    ```bash
    pip install numpy opencv-python PyQt5 matplotlib
    ```
2.  **Run the Application:**
    ```bash
    python main.py
    ```

## Technologies Used
* **Language:** Python 3.x
* **
