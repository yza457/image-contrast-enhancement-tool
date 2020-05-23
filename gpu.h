#ifndef __GPU_H_
#define __GPU_H_

#include "hist-equ.h"

void initialize_cuda();
PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in);
PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in);

#endif
