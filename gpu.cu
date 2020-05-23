#include "gpu.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <iostream>
#include <stdio.h>

#define NUM_BINS 256

__global__ void initialize_gpu() {
  // Do nothing
  return;
}

void initialize_cuda() {
  cudaSetDevice(0);
  initialize_gpu<<<1, 1>>>();
}

__device__ float gpu_Hue_2_RGB(float v1, float v2,
                               float vH) // Function Hue_2_RGB
{
  if ( vH < 0 ) vH += 1;
  if ( vH > 1 ) vH -= 1;
  if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
  if ( ( 2 * vH ) < 1 ) return ( v2 );
  if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
  return ( v1 );
}

__device__ unsigned char gpu_clip_rgb(int x) { 
  if(x > 255) return 255;
  if(x < 0) return 0;

  return (unsigned char)x;
}

// Convert RGB to HSL, assume R,G,B in [0, 255]
// Output H, S in [0.0, 1.0] and L in [0, 255]
__global__ void gpu_rgb2hsl(const unsigned char *R, const unsigned char *G,
                            const unsigned char *B, float *out_H, float *out_S,
                            unsigned char *out_L, int imgWidth, int imgHeight) {
  // global image coordinates
  int img_x = blockIdx.x * blockDim.x + threadIdx.x;
  int img_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (img_x < imgWidth && img_y < imgHeight) { // bounds check
    // global image linear index
    int img_idx = img_y * imgWidth + img_x;

    // initialize return values
    float H = 0.0;
    float S = 0.0;
    float L = 0.0;

    // read pixel value
    float var_r = ( (float)R[img_idx] / 255 );
    float var_g = ( (float)G[img_idx] / 255 );
    float var_b = ( (float)B[img_idx] / 255 );

    float var_min = (var_r < var_g) ? var_r : var_g;
    var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
    float var_max = (var_r > var_g) ? var_r : var_g;
    var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
    float del_max = var_max - var_min;               //Delta RGB value
    // calculate L_
    L = ( var_max + var_min ) / 2;
    if ( del_max == 0 )//This is a gray, no chroma...
    {
        H = 0;
        S = 0;
    }
    else                                    //Chromatic data...
    {
        if ( L < 0.5 )
            S = del_max/(var_max+var_min);
        else
            S = del_max/(2-var_max-var_min );

        float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
        float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
        float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
        if( var_r == var_max ) {
            H = del_b - del_g;
        }
        else {
            if( var_g == var_max ) {
                H = (1.0/3.0) + del_r - del_b;
            }
            else {
                H = (2.0/3.0) + del_g - del_r;
            }
        }

    }

    if ( H < 0 )
        H += 1;
    if ( H > 1 )
        H -= 1;
    // write to output locations
    out_H[img_idx] = H;
    out_S[img_idx] = S;
    out_L[img_idx] = (unsigned char) (L * 255);
  }

}

__global__ void gpu_hsl2rgb(unsigned char *out_R, unsigned char *out_G,
                            unsigned char *out_B, const float *in_H,
                            const float *in_S, const unsigned char *in_L,
                            int imgWidth, int imgHeight) {
    // global image coordinates
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (img_x < imgWidth && img_y < imgHeight) { // bounds check
      // global image linear index
      int img_idx = img_y * imgWidth + img_x;

      // initialize return values
      unsigned char r, g, b;

      // read input HSL value
      float H = in_H[img_idx];
      float S = in_S[img_idx];
      float L = in_L[img_idx] / 255.0f;
      float var_1, var_2;

      if (S == 0) {
        r = L * 255;
        g = L * 255;
        b = L * 255;
      } else {
        if ( L < 0.5 ) {
        var_2 = L * ( 1 + S );
        } else {
        var_2 = ( L + S ) - ( S * L );
        }  
        var_1 = 2 * L - var_2;
        r = 255 * gpu_Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
        g = 255 * gpu_Hue_2_RGB( var_1, var_2, H );
        b = 255 * gpu_Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
      }
      out_R[img_idx] = r;
      out_G[img_idx] = g;
      out_B[img_idx] = b;
    }
    
  }

// Generate a histogram
__global__ void gpu_create_histogram(int *hist_out, unsigned char *img_in,
                                     int imgWidth, int imgHeight) {
  // global image coordinates
  int img_x = blockIdx.x * blockDim.x + threadIdx.x;
  int img_y = blockIdx.y * blockDim.y + threadIdx.y;

  // linear thread index within block
  int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

  // number of threads in a block
  int nt = blockDim.x * blockDim.y;

  // local histogram in shared memory
  __shared__ int local_hist[NUM_BINS];

  // initialize local_hist
  for (int i = linear_tid; i < NUM_BINS; i += nt) local_hist[i] = 0;

  // wait for all threads to complete
  __syncthreads();

  // update local_hist
  if (img_x < imgWidth && img_y < imgHeight) {
    int pixel_val = img_in[img_y * imgWidth + img_x];
    atomicAdd(&local_hist[pixel_val], 1);
  }

  // wait for all threads to complete
  __syncthreads();

  // update hist_out
  for (int i = linear_tid; i < NUM_BINS; i += nt) atomicAdd(&hist_out[i], local_hist[i]);

}

// Equalize the image using the provided LUT
__global__ void gpu_histogram_equalization(unsigned char *img_out,
                                           unsigned char *img_in, int *lut,
                                           int imgWidth, int imgHeight) {
    // global image coordinates
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;

    // update img_out
    if (img_x < imgWidth && img_y < imgHeight) {
      int img_idx = img_y * imgWidth + img_x;
      unsigned char old_val = img_in[img_idx];
      if (lut[old_val] > 255) {
        img_out[img_idx] = (unsigned char) 255;
      } else {
        img_out[img_idx] = (unsigned char) lut[old_val];
      }
      // if (img_x == 0 && img_y == 0) {
      //   printf("img_idx is %d\n", img_idx);
      //   printf("old_val is %u\n", old_val);
      //   printf("new_val is %u\n", img_out[img_idx]);
      // }
    }
}

void gpu_perform_equalization(dim3 dimBlock, dim3 dimGrid,
                              unsigned char *input_channel,
                              unsigned char *output_channel, int imgWidth,
                              int imgHeight) {
  cudaError_t err = cudaSuccess;
  int *d_histogram = NULL;
  cudaError_t err_histogram =
      cudaMalloc((void **)&d_histogram, NUM_BINS * sizeof(int));
  if (err_histogram != cudaSuccess) {
    std::cerr << "Failed to allocate device memory" << std::endl;
    exit(EXIT_FAILURE);
  }
  err_histogram = cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));
  if (err_histogram != cudaSuccess) {
    std::cerr << "Failed to set histogram memory" << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  // Create the histogram
  gpu_create_histogram<<<dimGrid, dimBlock>>>(d_histogram, input_channel, imgWidth, imgHeight);
  cudaDeviceSynchronize();
  sdkStopTimer(&timer);
  printf("Histogram creation processing time: %f (ms)\n",
         sdkGetTimerValue(&timer));
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cerr << "Failed to perform histogram creation:" << std::endl
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  // Begin histogram equalization code by performing LUT creation on CPU
  // as it'll be much faster than on GPU
  int* h_lut = (int *) malloc(sizeof(int) * NUM_BINS);
  int* h_histogram = (int *) malloc(sizeof(int) * NUM_BINS);
  if (h_lut == NULL || h_histogram == NULL) {
    printf("malloc for h_lut or h_histogram failed\n");
    exit(-1);
  }
  cudaError_t err_histo_transfer = cudaMemcpy(h_histogram, d_histogram, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);
  if (err_histo_transfer != cudaSuccess) {
    std::cerr << "Failed to perform histogram transfer:" << std::endl
    << cudaGetErrorString(err_histo_transfer) << std::endl;
    exit(EXIT_FAILURE);
  }
  int cdf = 0;
  int min = 0;
  int i = 0;

  while(min == 0) {
      min = h_histogram[i++];
  }
  int d = imgWidth * imgHeight - min;
  for(i = 0; i < NUM_BINS; i++) {
    cdf += h_histogram[i];
    //h_lut[i] = (cdf - min)*(NUM_BINS - 1)/d;
    h_lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    if(h_lut[i] < 0) {
        h_lut[i] = 0;
    }
  }
  long long count = 0;
  for (int i = 0; i < NUM_BINS; i++) {
    count += h_histogram[i];
  }
  // if (count == imgWidth * imgHeight) {
  //   printf("HISTROGRAM IS CORRECT!\n");
  // } else {
  //   exit(1);
  // }
  // Perform the actual histogram equalization
  int *d_lut = NULL;
  cudaError_t err_lut_malloc = cudaMalloc((void **)&d_lut, NUM_BINS * sizeof(int));
  if (err_lut_malloc != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for lut" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaError_t err_lut_transfer = cudaMemcpy(d_lut, h_lut, sizeof(int) * NUM_BINS, cudaMemcpyHostToDevice);
  if (err_histogram != cudaSuccess) {
    std::cerr << "Failed to transfer lut to device" << std::endl;
    exit(EXIT_FAILURE);
  }
  gpu_histogram_equalization<<<dimGrid, dimBlock>>>(output_channel, input_channel, d_lut, imgWidth, imgHeight); // actual equalization
  cudaDeviceSynchronize();
  sdkStopTimer(&timer);
  printf("Histogram equalization processing time: %f (ms)\n",
         sdkGetTimerValue(&timer));
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cerr << "Failed to perform histogram equalization:" << std::endl
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  err_histogram = cudaFree(d_histogram);
  if (err_histogram != cudaSuccess) {
    std::cerr << "Failed to free device memory" << std::endl;
    exit(EXIT_FAILURE);
  }
  free(h_lut);
  free(h_histogram);
  cudaFree(d_lut);
}

PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in) {
  // cudaError_t err = cudaSuccess;

  int numElements = img_in.w * img_in.h;
  size_t size_char = numElements * sizeof(unsigned char);
  size_t size_float = numElements * sizeof(float);

  unsigned char *h_R = (unsigned char *)malloc(size_char);
  memcpy(h_R, img_in.img_r, size_char);
  unsigned char *h_G = (unsigned char *)malloc(size_char);
  memcpy(h_G, img_in.img_g, size_char);
  unsigned char *h_B = (unsigned char *)malloc(size_char);
  memcpy(h_B, img_in.img_b, size_char);

  unsigned char *d_R = NULL;
  cudaError_t err_R = cudaMalloc((void **)&d_R, size_char);
  unsigned char *d_G = NULL;
  cudaError_t err_G = cudaMalloc((void **)&d_G, size_char);
  unsigned char *d_B = NULL;
  cudaError_t err_B = cudaMalloc((void **)&d_B, size_char);
  float *d_H = NULL;
  cudaError_t err_H = cudaMalloc((void **)&d_H, size_float);
  float *d_S = NULL;
  cudaError_t err_S = cudaMalloc((void **)&d_S, size_float);
  unsigned char *d_L = NULL;
  cudaError_t err_L = cudaMalloc((void **)&d_L, size_char);
  unsigned char *d_result = NULL;
  cudaError_t err_result = cudaMalloc((void **)&d_result, size_char);

  if (err_R != cudaSuccess || err_G != cudaSuccess || err_B != cudaSuccess ||
      err_H != cudaSuccess || err_S != cudaSuccess || err_L != cudaSuccess ||
      err_result != cudaSuccess) {
    std::cerr << "Failed to allocate device memory" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Copy the RGB channels into GPU memory
  err_R = cudaMemcpy(d_R, h_R, size_char, cudaMemcpyHostToDevice);
  err_G = cudaMemcpy(d_G, h_G, size_char, cudaMemcpyHostToDevice);
  err_B = cudaMemcpy(d_B, h_B, size_char, cudaMemcpyHostToDevice);

  if (err_R != cudaSuccess || err_G != cudaSuccess || err_B != cudaSuccess) {
    std::cerr << "Failed to copy RGB to device" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Round up the nearest dimBlock.x/y size (in the event the image size
  // is not a dividable by 16)
  int thread = 16; // block size
  // define block and grid dimensions
  dim3 dimBlock(thread, thread);
  dim3 dimGrid( (img_in.w + thread -1) / thread, (img_in.h + thread - 1) / thread );
  // Perform rgb2hsl
  gpu_rgb2hsl<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_H, d_S, d_L, img_in.w, img_in.h);
  cudaDeviceSynchronize();
  // Perform equalization
  gpu_perform_equalization(dimBlock, dimGrid, d_L, d_result, img_in.w, img_in.h);
  // Perform hsl2rgb
  gpu_hsl2rgb<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_H, d_S, d_result, img_in.w, img_in.h);
  cudaDeviceSynchronize();
  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  err_R = cudaMemcpy(h_R, d_R, size_char, cudaMemcpyDeviceToHost);
  err_G = cudaMemcpy(h_G, d_G, size_char, cudaMemcpyDeviceToHost);
  err_B = cudaMemcpy(h_B, d_B, size_char, cudaMemcpyDeviceToHost);

  if (err_R != cudaSuccess || err_G != cudaSuccess || err_B != cudaSuccess) {
    std::cerr << "Failed to copy RGB to device" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_H);
  cudaFree(d_S);
  cudaFree(d_L);
  cudaFree(d_result);
  PPM_IMG result;
  result.w = img_in.w;
  result.h = img_in.h;
  result.img_r = h_R;
  result.img_g = h_G;
  result.img_b = h_B;
  return result;
}

__global__ void gpu_rgb2yuv(const unsigned char *in_R,
                            const unsigned char *in_G,
                            const unsigned char *in_B, unsigned char *out_Y,
                            unsigned char *out_U, unsigned char *out_V,
                            int img_width, int img_height) {
  // global image coordinates
  int img_x = blockIdx.x * blockDim.x + threadIdx.x;
  int img_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (img_x < img_width && img_y < img_height) { // bounds check
    // global image linear index
    int img_idx = img_y * img_width + img_x;

    unsigned char r = in_R[img_idx];
    unsigned char g = in_G[img_idx];
    unsigned char b = in_B[img_idx];

    unsigned char y = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    unsigned char cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    unsigned char cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);

    out_Y[img_idx] = y;
    out_U[img_idx] = cb;
    out_V[img_idx] = cr;
  }  
}
__global__ void gpu_yuv2rgb(const unsigned char *in_Y,
                            const unsigned char *in_U,
                            const unsigned char *in_V, unsigned char *out_R,
                            unsigned char *out_G, unsigned char *out_B,
                            int img_width, int img_height) {
    // global image coordinates
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (img_x < img_width && img_y < img_height) { // bounds check
      // global image linear index
      int img_idx = img_y * img_width + img_x;

      int y = (int)in_Y[img_idx];
      int cb = (int)in_U[img_idx] - 128;
      int cr = (int)in_V[img_idx] - 128;

      int rt  = (int)( y + 1.402*cr);
      int gt  = (int)( y - 0.344*cb - 0.714*cr);
      int bt  = (int)( y + 1.772*cb);

      out_R[img_idx] = gpu_clip_rgb(rt);
      out_G[img_idx] = gpu_clip_rgb(gt);
      out_B[img_idx] = gpu_clip_rgb(bt);
    }  
}

PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in) {

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in device memory
  int numElements = img_in.w * img_in.h;
  size_t size_char = numElements * sizeof(unsigned char);
  size_t size_float = numElements * sizeof(float);

  unsigned char *h_R = (unsigned char *)malloc(size_char);
  memcpy(h_R, img_in.img_r, size_char);
  unsigned char *h_G = (unsigned char *)malloc(size_char);
  memcpy(h_G, img_in.img_g, size_char);
  unsigned char *h_B = (unsigned char *)malloc(size_char);
  memcpy(h_B, img_in.img_b, size_char);

  unsigned char *d_R = NULL;
  cudaError_t err_R = cudaMalloc((void **)&d_R, size_char);
  unsigned char *d_G = NULL;
  cudaError_t err_G = cudaMalloc((void **)&d_G, size_char);
  unsigned char *d_B = NULL;
  cudaError_t err_B = cudaMalloc((void **)&d_B, size_char);
  unsigned char *d_Y = NULL;
  cudaError_t err_Y = cudaMalloc((void **)&d_Y, size_float);
  unsigned char *d_U = NULL;
  cudaError_t err_U = cudaMalloc((void **)&d_U, size_float);
  unsigned char *d_V = NULL;
  cudaError_t err_V = cudaMalloc((void **)&d_V, size_char);
  unsigned char *d_result = NULL;
  cudaError_t err_result = cudaMalloc((void **)&d_result, size_char);

  if (err_R != cudaSuccess || err_G != cudaSuccess || err_B != cudaSuccess ||
      err_Y != cudaSuccess || err_U != cudaSuccess || err_V != cudaSuccess ||
      err_result != cudaSuccess) {
    std::cerr << "Failed to allocate device memory" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Copy the RGB channels into GPU memory
  err_R = cudaMemcpy(d_R, h_R, size_char, cudaMemcpyHostToDevice);
  err_G = cudaMemcpy(d_G, h_G, size_char, cudaMemcpyHostToDevice);
  err_B = cudaMemcpy(d_B, h_B, size_char, cudaMemcpyHostToDevice);

  // Round up the nearest dimBlock.x/y size (in the event the image size
  // is not a dividable by 16)
  int thread = 16; // block size
  // define block and grid dimensions
  dim3 dimBlock(thread, thread);
  dim3 dimGrid( (img_in.w + thread -1) / thread, (img_in.h + thread - 1) / thread );
  // Perform rgb2yuv
  gpu_rgb2yuv<<<dimGrid, dimBlock>>>(d_R, d_G, d_B, d_Y, d_U, d_V, img_in.w, img_in.h);
  cudaDeviceSynchronize();
  // Perform equalization
  // in host memory.
  gpu_perform_equalization(dimBlock, dimGrid, d_Y, d_result, img_in.w, img_in.h);
  // Perform yuv2rgb
  gpu_yuv2rgb<<<dimGrid, dimBlock>>>(d_result, d_U, d_V, d_R, d_G, d_B, img_in.w, img_in.h);
  cudaDeviceSynchronize();
  // copy rbg back to host
  err_R = cudaMemcpy(h_R, d_R, size_char, cudaMemcpyDeviceToHost);
  err_G = cudaMemcpy(h_G, d_G, size_char, cudaMemcpyDeviceToHost);
  err_B = cudaMemcpy(h_B, d_B, size_char, cudaMemcpyDeviceToHost);

  if (err_R != cudaSuccess || err_G != cudaSuccess || err_B != cudaSuccess) {
    std::cerr << "Failed to copy RGB to device" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_Y);
  cudaFree(d_U);
  cudaFree(d_V);
  cudaFree(d_result);
  PPM_IMG result;
  result.w = img_in.w;
  result.h = img_in.h;
  result.img_r = h_R;
  result.img_g = h_G;
  result.img_b = h_B;
  return result;
}

PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in) {
  // Round up the nearest dimBlock.x/y size (in the event the image size
  // is not a dividable by 16)
  int imgW = img_in.w;
  int imgH = img_in.h;
  size_t bytes = imgW * imgH * sizeof(unsigned char);
  int thread = 16; // block size;

  // define block and grid dimensions
  dim3 dimBlock(thread, thread);
  dim3 dimGrid( (imgW + thread -1) / thread, (imgH + thread - 1) / thread ); // round up
  
  // allocate host memory for output and device memory for input and output
  unsigned char * h_input = img_in.img;
  unsigned char * h_output = (unsigned char *) malloc(bytes); // this will be freed by caller
  unsigned char * d_input;
  unsigned char * d_output;
  cudaMalloc((void **) &d_input, bytes);
  cudaMalloc((void **) &d_output, bytes);

  // copy h_input to d_input;
  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

  // Perform equalization
  gpu_perform_equalization(dimBlock, dimGrid, d_input, d_output, imgW, imgH);

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  // return new image
  PGM_IMG result;
  result.w = img_in.w;
  result.h = img_in.h;
  result.img = h_output;
  return result;
}
