#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Image {
  int width;
  int height;
  std::vector<unsigned char> pixels;
};

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                << " at line " << __LINE__ << std::endl;      \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

bool ReadPpm(const std::string& filename, Image* image) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) return false;

  std::string magic;
  file >> magic;
  if (magic != "P6") return false;

  file >> image->width >> image->height;

  int max_value;
  file >> max_value;
  file.ignore(1);

  int size = image->width * image->height * 3;
  image->pixels.resize(size);
  file.read(reinterpret_cast<char*>(image->pixels.data()), size);

  return true;
}

bool WritePpm(const std::string& filename, const Image& image) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return false;

  file << "P6\n" << image.width << " " << image.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(image.pixels.data()),
             image.pixels.size());

  return true;
}

__global__ void GrayscaleKernel(const unsigned char* input,
                                unsigned char* output,
                                int width,
                                int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;

  if (idx >= total_pixels) return;

  int rgb_idx = idx * 3;
  unsigned char r = input[rgb_idx];
  unsigned char g = input[rgb_idx + 1];
  unsigned char b = input[rgb_idx + 2];

  unsigned char gray =
      static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

  output[rgb_idx] = gray;
  output[rgb_idx + 1] = gray;
  output[rgb_idx + 2] = gray;
}

__global__ void BlurKernel(const unsigned char* input,
                           unsigned char* output,
                           int width,
                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  for (int c = 0; c < 3; ++c) {
    int sum = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = x + dx;
        int ny = y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int index = (ny * width + nx) * 3 + c;
          sum += input[index];
          ++count;
        }
      }
    }

    int out_index = (y * width + x) * 3 + c;
    output[out_index] = static_cast<unsigned char>(sum / count);
  }
}

void ProcessImageOnGpu(const Image& input_image, Image* output_image) {
  int size = input_image.width * input_image.height * 3;

  output_image->width = input_image.width;
  output_image->height = input_image.height;
  output_image->pixels.resize(size);

  unsigned char* d_input = nullptr;
  unsigned char* d_gray = nullptr;
  unsigned char* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, size));
  CUDA_CHECK(cudaMalloc(&d_gray, size));
  CUDA_CHECK(cudaMalloc(&d_output, size));

  CUDA_CHECK(cudaMemcpy(d_input,
                        input_image.pixels.data(),
                        size,
                        cudaMemcpyHostToDevice));

  int total_pixels = input_image.width * input_image.height;
  int threads = 256;
  int blocks = (total_pixels + threads - 1) / threads;

  GrayscaleKernel<<<blocks, threads>>>(
      d_input, d_gray, input_image.width, input_image.height);
  CUDA_CHECK(cudaGetLastError());

  dim3 block_dim(16, 16);
  dim3 grid_dim((input_image.width + block_dim.x - 1) / block_dim.x,
                (input_image.height + block_dim.y - 1) / block_dim.y);

  BlurKernel<<<grid_dim, block_dim>>>(
      d_gray, d_output, input_image.width, input_image.height);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output_image->pixels.data(),
                        d_output,
                        size,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gray));
  CUDA_CHECK(cudaFree(d_output));
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_directory> <output_directory>\n";
    return EXIT_FAILURE;
  }

  std::string input_dir = argv[1];
  std::string output_dir = argv[2];

  fs::create_directories(output_dir);
  fs::create_directories("artifacts");

  std::ofstream log_file("artifacts/execution_log.txt");
  log_file << "CUDA Batch Image Processing Pipeline\n";
  log_file << "Input directory: " << input_dir << "\n";
  log_file << "Output directory: " << output_dir << "\n\n";

  int processed_count = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.path().extension() != ".ppm") {
      continue;
    }

    Image input_image;
    if (!ReadPpm(entry.path().string(), &input_image)) {
      std::cerr << "Skipping invalid image: " << entry.path() << "\n";
      continue;
    }

    Image output_image;
    ProcessImageOnGpu(input_image, &output_image);

    std::string output_filename =
        output_dir + "/processed_" + entry.path().filename().string();

    WritePpm(output_filename, output_image);

    log_file << "Processed: " << entry.path().filename().string()
             << " -> " << output_filename
             << " (" << input_image.width << "x" << input_image.height
             << ")\n";

    ++processed_count;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  log_file << "\nTotal images processed: " << processed_count << "\n";
  log_file << "Total execution time: " << elapsed_ms << " ms\n";

  std::cout << "Processed " << processed_count << " images.\n";
  std::cout << "Execution log saved to artifacts/execution_log.txt\n";

  return EXIT_SUCCESS;
}