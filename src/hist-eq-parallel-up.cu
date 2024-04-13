#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stb_image_write.h"

#define COLOR_CHANNELS 0
#define COLORLEVELS 256
#define BLOCK_SIZE_HIST 1024
#define BLOCK_SIZE_EQ 1024

struct {
    size_t width;
    size_t height;
    size_t channels;
} image_data;

/**
 * @brief Checks a condition and prints an error message if the condition is false.
 * 
 * @param condition The condition to check.
 * @param message The error message to print.
 */
void check_and_print_error(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        exit(1);
    }
}

/**
 * @brief Computes the index of a pixel in the image.
 * 
 * @param x The x coordinate of the pixel.
 * @param y The y coordinate of the pixel.
 * @param channel The color channel of the pixel.
 * @param width The width of the image.
 * @param channels The number of channels in the image.
 * @return size_t The index of the pixel in the image.
 */
__device__ size_t index(size_t x, size_t y, int channel, int width, int channels) {
    return y * width * channels + channel + x * channels;
}

/**
 * @brief Finds the non-zero minimum in the cumulative distribution function.
 * 
 * @param cdf The cumulative distribution function.
 * @return uint32_t The non-zero minimum in the cumulative distribution function. 
 */
uint32_t cdf_find_min(uint32_t *cdf) {
    uint32_t min = 0;
    for (size_t i = 0; i < COLORLEVELS; i++) {
        if (cdf[i] != 0) {
            min = cdf[i];
            break;
        }
    }
    return min;
}

/**
 * @brief Calculates the new pixel intensity using the equation: floor(((cdf - cdf_min) * (COLORLEVELS - 1)) / (total_pixels - cdf_min)).
 * 
 * @param cdf The cumulative distribution function.
 * @param cdf_min The non-zero minimum in the cumulative distribution function.
 * @param total_pixels The total number of pixels in the image.
 * @return uint8_t The new pixel intensity.
 */
__device__ uint8_t calculate_pixel_intensity(uint32_t cdf, uint32_t cdf_min, uint32_t total_pixels) {
    return (uint8_t)(((cdf - cdf_min) * (COLORLEVELS - 1)) / (total_pixels - cdf_min));
}

/**
 * @brief Computes the histogram of the image for a specific color channel.
 * 
 * @param image The image data.
 * @param histogram The histogram of the image.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param color_channel The color channel for which the histogram is to be computed.
 */
__global__ void compute_histogram(uint8_t *image, uint32_t *histogram, size_t width, size_t height, size_t color_channel, size_t cpp) {
    /*extern __shared__ uint32_t histogram_shared[COLORLEVELS];

    // Clear shared memory.
    if (threadIdx.x < COLORLEVELS) {
        histogram_shared[threadIdx.x] = 0;
    }

    __syncthreads();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (threadId < (width * height * cpp) / cpp) {
        atomicAdd(&histogram_shared[image[threadId * cpp + color_channel]], 1);
        threadId += stride;
    }

    __syncthreads();

    // Accumulate the shared histogram to the global histogram.
    if (threadIdx.x < COLORLEVELS) {
        atomicAdd(&histogram[threadIdx.x], histogram_shared[threadIdx.x]);
    }*/

    // Initialize the partial histograms.
    extern __shared__ unsigned int partial_histogram[COLORLEVELS];
    for (size_t i = threadIdx.x; i < COLORLEVELS; i += blockDim.x) {
        partial_histogram[i] = 0;
    }

    __syncthreads();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (threadId < width * height) {
        int pixel_value = image[threadId * cpp + color_channel];
        atomicAdd(&partial_histogram[pixel_value], 1);
        threadId += stride;
    }

    __syncthreads();

    // Accumulate the partial histograms to the global histogram.
    for (size_t i = threadIdx.x; i < COLORLEVELS; i += blockDim.x) {
        atomicAdd(&histogram[i], partial_histogram[i]);
    }
}

/**
 * @brief Computes the cumulative distribution function of the histogram.
 * 
 * @param histogram The histogram of the image.
 * @param cdf The cumulative distribution function.
 */
__global__ void compute_cdf(uint32_t *histogram, uint32_t *cdf) {
    extern __shared__ float scan[2 * COLORLEVELS];

    int threadId = threadIdx.x;
    int offset = 1;

    scan[2 * threadId] = histogram[2 * threadId];
    scan[2 * threadId + 1] = histogram[2 * threadId + 1];

    // Build sum in place.
    for (int d = COLORLEVELS >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            scan[bi] += scan[ai];
        }

        offset *= 2;
    }

    if (threadId == 0) {
        scan[COLORLEVELS - 1] = 0;
    }

    // Traverse down tree and build scan.
    for (int d = 1; d < COLORLEVELS; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;

            float t = scan[ai];
            scan[ai] = scan[bi];
            scan[bi] += t;
        }
    }
    __syncthreads();

    // Store results of scan into cumulative distribution function.
    cdf[2 * threadId] = scan[2 * threadId];
    cdf[2 * threadId + 1] = scan[2 * threadId + 1];
}

/**
 * Equalizes the color image by equalizing each color channel.
 * 
 * @param image_in The image input parameter.
 * @param image_out The image output parameter
 * @param width The width of the image.
 * @param height The height of the image. 
 */
__global__ void equalize_image(uint8_t *image_in, uint8_t *image_out, size_t width, size_t height, 
                            uint32_t *cdf_red, uint32_t *cdf_green, uint32_t *cdf_blue,
                            uint32_t cdf_min_red, uint32_t cdf_min_green, uint32_t cdf_min_blue) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int imageSize = width * height * 3;

    while (threadId < imageSize) {
        image_out[threadId] = calculate_pixel_intensity(cdf_red[image_in[threadId]], cdf_min_red, width * height);
        image_out[threadId + 1] = calculate_pixel_intensity(cdf_green[image_in[threadId + 1]], cdf_min_green, width * height);
        image_out[threadId + 2] = calculate_pixel_intensity(cdf_blue[image_in[threadId + 2]], cdf_min_blue, width * height);
        threadId += stride;
    }
    
}

int main(int argc, char *argv[]) {
    check_and_print_error(argc == 3, "USAGE: sample input_image output_image");

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image.
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    image_data.width = width;
    image_data.height = height;
    image_data.channels = cpp;

    char error_message[300];
    snprintf(error_message, 300, "Error reading loading image %s!", szImage_in_name);
    check_and_print_error(h_imageIn != NULL, error_message);

    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *) malloc (datasize);

    // Setup Thread organization
    dim3 block_size_hist(BLOCK_SIZE_HIST);
    dim3 grid_size_hist((width * height * 3 - 1) / block_size_hist.x + 1);

    dim3 block_size_cdf(256); // 256 threads per block - 256 color levels.
    dim3 grid_size_cdf(1); 

    dim3 block_size_eq(BLOCK_SIZE_EQ);
    dim3 grid_size_eq((width * height * 3 - 1) / block_size_eq.x + 1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    uint32_t *histogram_red;
    uint32_t *histogram_green;
    uint32_t *histogram_blue;

    uint32_t *cdf_red;
    uint32_t *cdf_green;
    uint32_t *cdf_blue;

    // Allocate memory on the device.
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));
    checkCudaErrors(cudaMalloc(&histogram_red, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&histogram_green, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&histogram_blue, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&cdf_red, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&cdf_green, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(&cdf_blue, COLORLEVELS * sizeof(uint32_t)));

    // Set the histograms and cdfs to zero.
    checkCudaErrors(cudaMemset(histogram_red, 0, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(histogram_green, 0, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(histogram_blue, 0, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(cdf_red, 0, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(cdf_green, 0, COLORLEVELS * sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(cdf_blue, 0, COLORLEVELS * sizeof(uint32_t)));

    // Use CUDA events to measure execution time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t start_hist, stop_hist;
    cudaEventCreate(&start_hist);
    cudaEventCreate(&stop_hist);

    cudaEvent_t start_cdf, stop_cdf;
    cudaEventCreate(&start_cdf);
    cudaEventCreate(&stop_cdf);

    cudaEvent_t start_cdf_min, stop_cdf_min;
    cudaEventCreate(&start_cdf_min);
    cudaEventCreate(&stop_cdf_min);

    cudaEvent_t start_eq, stop_eq;
    cudaEventCreate(&start_eq);
    cudaEventCreate(&stop_eq);

    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));

    // Copy image to device and run kernel
    cudaEventRecord(start);

    cudaEventRecord(start_hist);

    compute_histogram<<<grid_size_hist, block_size_hist>>>(d_imageIn, histogram_red, width, height, 0, cpp);
    compute_histogram<<<grid_size_hist, block_size_hist>>>(d_imageIn, histogram_green, width, height, 1, cpp);
    compute_histogram<<<grid_size_hist, block_size_hist>>>(d_imageIn, histogram_blue, width, height, 2, cpp);

    cudaEventRecord(stop_hist);
    cudaEventSynchronize(stop_hist);

    cudaEventRecord(start_cdf);

    compute_cdf<<<grid_size_cdf, block_size_cdf>>>(histogram_red, cdf_red);
    compute_cdf<<<grid_size_cdf, block_size_cdf>>>(histogram_green, cdf_green);
    compute_cdf<<<grid_size_cdf, block_size_cdf>>>(histogram_blue, cdf_blue);

    cudaEventRecord(stop_cdf);
    cudaEventSynchronize(stop_cdf);

    uint32_t h_cdf_red[COLORLEVELS];
    uint32_t h_cdf_green[COLORLEVELS];
    uint32_t h_cdf_blue[COLORLEVELS];

    cudaEventRecord(start_cdf_min);

    checkCudaErrors(cudaMemcpy(h_cdf_red, cdf_red, COLORLEVELS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_cdf_green, cdf_green, COLORLEVELS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_cdf_blue, cdf_blue, COLORLEVELS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    uint32_t cdf_min_red = cdf_find_min(h_cdf_red);
    uint32_t cdf_min_green = cdf_find_min(h_cdf_green);
    uint32_t cdf_min_blue = cdf_find_min(h_cdf_blue);

    cudaEventRecord(stop_cdf_min);
    cudaEventSynchronize(stop_cdf_min);

    cudaEventRecord(start_eq);

    equalize_image<<<grid_size_eq, block_size_eq>>>(
        d_imageIn, d_imageOut, width, height, 
        cdf_red, cdf_green, cdf_blue,
        cdf_min_red, cdf_min_green, cdf_min_blue
    );

    cudaEventRecord(stop_eq);
    cudaEventSynchronize(stop_eq);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));
    getLastCudaError("copy_image() execution failed\n");



    // Print time.
    float milliseconds = 0;
    
    cudaEventElapsedTime(&milliseconds, start_hist, stop_hist);
    printf("Histogram Computation time is: %0.3f milliseconds \n", milliseconds);

    cudaEventElapsedTime(&milliseconds, start_cdf, stop_cdf);
    printf("CDF Computation time is: %0.3f milliseconds \n", milliseconds);

    cudaEventElapsedTime(&milliseconds, start_cdf_min, stop_cdf_min);
    printf("CDF Min Computation time is: %0.3f milliseconds \n", milliseconds);

    cudaEventElapsedTime(&milliseconds, start_eq, stop_eq);
    printf("Equalization Computation time is: %0.3f milliseconds \n", milliseconds);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Write the output file.
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Free device memory.
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(histogram_red));
    checkCudaErrors(cudaFree(histogram_green));
    checkCudaErrors(cudaFree(histogram_blue));
    checkCudaErrors(cudaFree(cdf_red));
    checkCudaErrors(cudaFree(cdf_green));
    checkCudaErrors(cudaFree(cdf_blue));

    // Clean-up events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_hist);
    cudaEventDestroy(stop_hist);
    cudaEventDestroy(start_cdf);
    cudaEventDestroy(stop_cdf);
    cudaEventDestroy(start_cdf_min);
    cudaEventDestroy(stop_cdf_min);
    cudaEventDestroy(start_eq);
    cudaEventDestroy(stop_eq);

    // Free host memory.
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
