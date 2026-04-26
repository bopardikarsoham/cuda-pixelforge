# CUDA PixelForge 🚀  
GPU-Accelerated Batch Image Processing Pipeline

---

## 📌 Overview

This project implements a **CUDA-based batch image processing pipeline** that processes a large number of images efficiently using GPU parallelism.

The program applies the following operations:

- Grayscale conversion  
- Blur filtering  

It processes **100+ images in a single execution**, demonstrating the advantage of GPU acceleration over CPU-based approaches.

---

## ⚙️ Features

- 🚀 GPU-accelerated computation using CUDA kernels  
- 📦 Batch processing of large image datasets  
- 🎯 Grayscale and blur filtering  
- 🧾 Execution logging for performance tracking  
- 🔧 Command-line interface (CLI) support  

---

## 🧠 Technologies Used

- CUDA (NVIDIA GPU Computing)
- C++
- Parallel Programming (CUDA Kernels)

---

## 📂 Project Structure

```
cuda-pixelforge/
│
├── src/                        # CUDA source code
│   └── batch_image_processor.cu
│
├── input/                      # Input images (.ppm format)
├── output/                     # Processed images
├── artifacts/                  # Logs and sample outputs
│   ├── execution_log.txt
│   └── sample_outputs/
│
├── Makefile                   # Build instructions
├── run.sh                     # Run script
├── README.md
```

---

## 🛠️ Build Instructions

```bash
make build
```

---

## ▶️ Run Instructions

```bash
./batch_processor input output
```

or

```bash
./run.sh
```

---

## 📊 Output

After execution:

- Processed images are saved in `output/`
- Execution logs are saved in `artifacts/execution_log.txt`

Example log:

```
Total images processed: 120
Total execution time: XXXX ms
```

---

## 📸 Proof of Execution

- 120 images processed in a single run  
- Before/after comparisons available in `artifacts/sample_outputs/`  
- Execution logs included  

---

## ⚠️ Requirements

- NVIDIA GPU  
- CUDA Toolkit  

> This project was executed in a GPU-enabled environment (Google Colab / Coursera CUDA Lab).

---

## 🧩 Challenges & Learnings

- Managing memory transfer between host and device  
- Optimizing CUDA kernel execution  
- Handling large-scale batch image processing  
- Understanding GPU parallelism and performance trade-offs  

---

## 🚀 Conclusion

This project demonstrates how CUDA-based GPU acceleration significantly improves performance for large-scale image processing tasks and highlights practical applications of parallel computing.

---

## 👤 Author

Anisha Bopardikar
