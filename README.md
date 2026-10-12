# OpenCV Document Scanner

Academic project demonstrating a classical computer vision pipeline for automated document scanning using **OpenCV** and **NumPy** without relying on deep learning.

## Pipeline Steps
1. **Preprocessing**: Grayscale conversion and Gaussian blur ($\text{kernel} = 7 \times 7$) for noise reduction.
2. **Binarization & Edge Detection**: Otsu's thresholding combined with contour retrieval (`findContours` + `approxPolyDP`) to isolate the document boundary (4-corner polygon).
3. **Perspective Correction**: Four-point geometric transform (`warpPerspective`) enforcing standard A4 aspect ratios ($1 : 1.414$).
4. **Post-processing**: Adaptive thresholding (`adaptiveThreshold`) to produce clean, high-contrast BW scan results (clearing shadows and background textures).

##  Performance Benchmarks (Execution Time)
Measured on standard CPU execution:
- Preprocessing: ~12-15 ms
- Contour Detection & Otsu: ~25-30 ms
- Perspective Warping (A4): ~15-20 ms
- Adaptive Thresholding: ~10-15 ms
- **Total Pipeline Execution Time: ~65–80 ms**

##  Requirements
Install dependencies via pip:
```bash
pip install -r requirements.txt
