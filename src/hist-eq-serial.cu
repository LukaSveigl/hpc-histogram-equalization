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
size_t index(size_t x, size_t y, int channel, int width, int channels) {
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
uint8_t calculate_pixel_intensity(uint32_t cdf, uint32_t cdf_min, uint32_t total_pixels) {
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
void compute_histogram(uint8_t *image, uint32_t *histogram, size_t width, size_t height, size_t color_channel) {
    // Empty the histogram.
    for (size_t i = 0; i < COLORLEVELS; i++) {
        histogram[i] = 0;
    }

    // Calculate histogram for the specific color channel.
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            histogram[image[index(j, i, color_channel, width, image_data.channels)]]++;
        }
    }
}

/**
 * @brief Computes the cumulative distribution function of the histogram.
 * 
 * @param histogram The histogram of the image.
 * @param cdf The cumulative distribution function.
 */
void compute_cdf(uint32_t *histogram, uint32_t *cdf) {
    cdf[0] = histogram[0];
    for (size_t i = 1; i < COLORLEVELS; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

/**
 * Equalizes the color image by equalizing each color channel.
 * 
 * @param image_in The image input parameter.
 * @param image_out The image output parameter
 * @param width The width of the image.
 * @param height The height of the image. 
 */
void equalize_image(uint8_t *image_in, uint8_t *image_out, size_t width, size_t height) {
    uint32_t *histogram = (uint32_t *)malloc(COLORLEVELS * sizeof(uint32_t));
    uint32_t *cdf = (uint32_t *)malloc(COLORLEVELS * sizeof(uint32_t));

    for (size_t i = 0; i < image_data.channels; i++) {
        compute_histogram(image_in, histogram, width, height, i);
        compute_cdf(histogram, cdf);
        uint32_t cdf_min = cdf_find_min(cdf);
        for (size_t j = 0; j < height; j++) {
            for (size_t k = 0; k < width; k++) {
                image_out[index(k, j, i, width, image_data.channels)] = calculate_pixel_intensity(cdf[image_in[index(k, j, i, width, image_data.channels)]], cdf_min, width * height);
            }
        }
    }

    free(histogram);
    free(cdf);
}

int main(int argc, char *argv[]) {
    check_and_print_error(argc == 3, "USAGE: sample input_image output_image");

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
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

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copy image to device and run kernel
    cudaEventRecord(start);
    
    equalize_image((uint8_t *)h_imageIn, (uint8_t *)h_imageOut, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Write the output file
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

    // Clean-up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
