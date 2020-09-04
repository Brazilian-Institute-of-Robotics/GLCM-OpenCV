/*
 * Copyright (c) 2019, SENAI Cimatec
 */


#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif


#define SRC_TSIZE 1 * (int)sizeof(srcT1)
#define DST_TSIZE 1 * (int)sizeof(dstT1)

#define noconvert

inline void atomicAdd_g_f(__local float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = current.f32 + val;
        current.u32 = atomic_cmpxchg( (__local unsigned int *)addr,
        expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

__kernel void glcm(__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols
                    )
{
    int x = get_global_id(0) / window;
    int y = get_global_id(1) * rowsPerWI / window * cn;
    int x_window = get_local_id(0);
    int y_window = get_local_id(1);

    int graycomatrix_local_index = mad24(x_window, window, y_window);

    __local float energy;
    energy = 0.0f;

#ifdef USE_GRAYLEVEL
    __local unsigned int gray_coordinates_int[n_level * n_level / 4];
    __local uchar *gray_coordinates = (__local uchar *) gray_coordinates_int;
#else
    __local unsigned int window_gray_coordinates[window*window];
    __local unsigned int window_gray_intensities[window*window];
#endif 
    __local unsigned int level_sum_loc;
    level_sum_loc = 0;

#ifdef USE_GRAYLEVEL
    int grayLevelsPerLocalInt = (grayLevelsPerLocal + cn) / 4;
    for(int i = 0; i < grayLevelsPerLocalInt; i++) {
        int graycomatrix_index = mad24(graycomatrix_local_index, grayLevelsPerLocalInt, i);
        gray_coordinates_int[graycomatrix_index] = 0;
    }
#else
    for (int c = 0; c < cn; ++c)
    {
        int y_window_cn = mad24(y_window, cn, c);
        int curr_window_gray_index = mad24(x_window, window, y_window_cn);
        window_gray_intensities[curr_window_gray_index] = 0;
        window_gray_coordinates[curr_window_gray_index] = 0;

    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < dst_cols)
    {
        int dst_index = mad24(y, dst_step, mad24(x, DST_TSIZE, dst_offset));

            if (y < dst_rows)
            {
                // __global const srcT1 * src = (__global const srcT1 *)(srcptr + src_index);
                __global dstT1 * dst = (__global dstT1 *)(dstptr + dst_index);
#pragma unroll
                for (int c = 0; c < cn; ++c)
                {
                    int y_window_cn =  mad24(y_window, cn, c);

                    int src_index = mad24(y + y_window_cn, src_step,mad24( x + x_window, SRC_TSIZE, src_offset));
                    __global const srcT1 * src = (__global const srcT1 *)(srcptr + src_index);
                    unsigned int gray_index = mad24(*src, n_level, *(src + window));

#ifdef USE_GRAYLEVEL

                    int shift_bits;
                    if(gray_index % 4 == 0) {
                        shift_bits = 0;
                    } else if( gray_index % 4 == 1 ) {
                        shift_bits = 8;
                    } else if( gray_index % 4 == 2 ) {
                        shift_bits = 16;
                    } else {
                        shift_bits = 24;
                    }
                    
                    atomic_add( &gray_coordinates_int[gray_index / 4], 1<<shift_bits);
#else
                    for(int i_window_gray = 0; i_window_gray < window * window; i_window_gray++) {
                        if(window_gray_coordinates[i_window_gray] == 0) {
                            int current_intensity = atomic_cmpxchg( &window_gray_coordinates[i_window_gray], 0,
                                gray_index + 1);
                            if(current_intensity == 0) {
                                atomic_add(&window_gray_intensities[i_window_gray], 1);
                                break;
                            } else {
                                if(current_intensity == gray_index + 1) {
                                    atomic_add(&window_gray_intensities[i_window_gray], 1);
                                    break;
                                } else {
                                    continue;
                                }

                            }
                        }
                        if(window_gray_coordinates[i_window_gray] == (gray_index + 1)) {
                            atomic_add(&window_gray_intensities[i_window_gray], 1);
                            break;
                        }
                    }
#endif
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
#ifdef USE_GRAYLEVEL
                for(int glcm_i = 0; glcm_i < grayLevelsPerLocalInt; glcm_i++) {
                    int graycomatrix_index = mad24(graycomatrix_local_index, grayLevelsPerLocalInt, glcm_i);
                    float current_energy = 0.0f;
                    current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 0 & 255) *
                        (float)(gray_coordinates_int[graycomatrix_index] >> 0 & 255) / (float)(window* window * window * window);
                    current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 8 & 255) *
                        (float)(gray_coordinates_int[graycomatrix_index] >> 8 & 255) / (float)(window* window * window * window);
                    current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 16 & 255) *
                        (float)(gray_coordinates_int[graycomatrix_index] >> 16 & 255) / (float)(window* window * window * window);
                    current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 24 & 255) *
                        (float)(gray_coordinates_int[graycomatrix_index] >> 24 & 255) / (float)(window* window * window * window);
                    // current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 0 & 255);
                    // current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 8 & 255);
                    // current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 16 & 255);
                    // current_energy += (float)(gray_coordinates_int[graycomatrix_index] >> 24 & 255);
                    
                    atomicAdd_g_f(&energy, current_energy);
                }
#else
                for (int c = 0; c < cn; ++c)
                {
                    int y_window_cn = mad24(y_window, cn, c);
                    int curr_window_gray_index = mad24(x_window, window, y_window_cn);
                    float current_energy = (float) (window_gray_intensities[curr_window_gray_index] * window_gray_intensities[curr_window_gray_index]) /
                            (float)(window* window * window * window);
                    // float current_energy = window_gray_intensities[curr_window_gray_index];
                    atomicAdd_g_f(&energy, current_energy);
                }
#endif

                barrier(CLK_LOCAL_MEM_FENCE);

                if(x_window == 0 && y_window == 0)
                    dst[0] = convertToDT(sqrt(energy));

                dst_index += dst_step;
                ++y;
            }
    }
}
