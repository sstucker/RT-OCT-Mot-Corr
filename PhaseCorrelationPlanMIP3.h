#pragma once
#include "fftw3.h"
#include <complex>
#include "Utils.h"


class PhaseCorrelationPlanMIP3
{
    protected:

        int* dims;
        int* og_dims;
        int fsize;
        int og_fsize;

        bool reference_acquired;  // true if a reference frame exists

        fftwf_complex* mip_buffer;

        fftwf_complex** mip_t0;
        fftwf_complex** mip_tn;
        fftwf_complex** mip_r;
        fftwf_complex** mip_R;

        fftwf_plan pc_mip_2d_fft_plan;
        fftwf_plan pc_mip_r_fft_plan;

        // Buffers for correlation calc
        std::complex<float> i_t0;
        std::complex<float> i_tn;
        std::complex<float> pc;
        std::complex<float> pcnorm;

        float* spectral_filter;
        float* spatial_filter; // Pixmap of attenuation values multiplied by the 2D spatial images prior to forward FFT 

        // Maxima finding buffers
        int maxi;
        int maxj;
        int maxk;

        int npeak;
        int upsample_factor;

        double dx;
        double dy;
        double dz;

        double centroid_i;
        double centroid_j;
        double centroid_k;
        double centroid_norm;
        std::complex<float>* maxima;

        float maxval;
        float mag;
        std::complex<float> tmp;
        double** fftshift;  // LUT of pixel shifts

        bool bidirectional;

        template <class T>
        inline T* indexBufferOG(T* buf, int i, int j, int k)
        {
            return buf + (k + og_dims[2] * (j + og_dims[1] * i));
        }

        template <class T>
        inline T* indexBuffer(T* buf, int i, int j, int k)
        {
            return buf + (k + dims[2] * (j + dims[1] * i));
        }

        template <class T>
        inline T* indexBuffer(T* buf, int i, int j)
        {
            return buf + (i * dims[0] + j);
        }

        inline void calculateMIP(fftwf_complex* result, fftwf_complex* src_3d, int axis)
        {
            memset(result, 0, og_dims[0] * og_dims[1] * sizeof(fftwf_complex));
            for (int i = 0; i < og_dims[0]; i++)
            {
                for (int j = 0; j < og_dims[1]; j++)
                {
                    float max = -1;
                    for (int k = 0; k < og_dims[0]; k++)  // MIP axis
                    {
                        if (axis == 0) { memcpy(&tmp, indexBufferOG(src_3d, k, i, j), sizeof(fftwf_complex)); }
                        else if (axis == 1) { memcpy(&tmp, indexBufferOG(src_3d, i, k, j), sizeof(fftwf_complex)); }
                        else { memcpy(&tmp, indexBufferOG(src_3d, i, j, k), sizeof(fftwf_complex)); }
                        
                        // Find max
                        mag = std::abs(tmp);
                        if (mag > max)
                        {
                            memcpy(result + (i * og_dims[0] + j), &tmp, sizeof(fftwf_complex));
                        }
                    }
                }
            }
        }

        inline void applyWindow2D(fftwf_complex* img, float* window)
        {
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    // Complex number
                    *(indexBuffer<fftwf_complex>(img, i, j))[0] *= *indexBuffer<float>(window, i, j);
                    *(indexBuffer<fftwf_complex>(img, i, j))[1] *= *indexBuffer<float>(window, i, j);
                }
            }
        }

        inline void populateZeroPaddedBuffer2D(fftwf_complex* dst, fftwf_complex* src)
        {
            // Copy from src buffer with dimensions i, j into center of zero-padded buffer

            memset(dst, 0, dims[0] * dims[1] * sizeof(fftwf_complex));  // Set the destination to 0, as it is zero-padded

            for (int i = 0; i < og_dims[0]; i++)
            {
                // TODO implement bidirectional reshape
                int dst_i = (dims[0] - og_dims[0]) / 2 + i;
                memcpy(dst + (dst_i * dims[0]), src + (i * og_dims[0]), sizeof(fftwf_complex) * og_dims[1]);
            }
        }

    public:

        PhaseCorrelationPlanMIP3()
        {
            fsize = 0;
            reference_acquired = false;

            dims = new int[3];
            og_dims = new int[3];
            memset(dims, 0, 3 * sizeof(int));
            memset(og_dims, 0, 3 * sizeof(int));
        }

        PhaseCorrelationPlanMIP3(int* input_dims, int upsample, int npeak_centroid, float* spectral_filter_2d, float* spatial_filter_2d, bool bidirectional)
        {

            reference_acquired = false;

            this->bidirectional = bidirectional;

            npeak = npeak_centroid;
            upsample_factor = upsample;

            dims = new int[3];
            og_dims = new int[3];
            memcpy(dims, input_dims, 3 * sizeof(int));
            memcpy(og_dims, input_dims, 3 * sizeof(int));

            fsize = 1;
            og_fsize = 1;
            for (int i = 0; i < 3; i++)
            {
                og_fsize *= og_dims[i];
                dims[i] *= upsample_factor;
                fsize *= dims[i];
            }

            spatial_filter = new float[dims[0] * dims[1]];
            spectral_filter = new float[dims[0] * dims[1]];

            memcpy(spatial_filter, spatial_filter_2d, dims[0] * dims[1] * sizeof(float));
            memcpy(spectral_filter, spectral_filter_2d, dims[0] * dims[1] * sizeof(float));

            // Assume that volume is square such that dim[0] and dim[1] describe each MIP
            mip_buffer = fftwf_alloc_complex(og_dims[0] * og_dims[1]);
            
            mip_t0 = new fftwf_complex * [3];
            mip_tn = new fftwf_complex * [3];
            mip_r = new fftwf_complex * [3];
            mip_R = new fftwf_complex * [3];
            for (int i = 0; i < 3; i++)
            {
                mip_t0[i] = fftwf_alloc_complex(dims[0] * dims[1]);
                mip_tn[i] = fftwf_alloc_complex(dims[0] * dims[1]);
                mip_r[i] = fftwf_alloc_complex(dims[0] * dims[1]);
                mip_R[i] = fftwf_alloc_complex(dims[0] * dims[1]);
            }

            fftwf_import_wisdom_from_filename("phasecorr_fftw_wisdom.txt");

            pc_mip_2d_fft_plan = fftwf_plan_dft_2d(dims[0], dims[1], mip_t0[0], mip_tn[0], FFTW_FORWARD, FFTW_PATIENT);
            pc_mip_r_fft_plan = fftwf_plan_dft_2d(dims[0], dims[1], mip_r[0], mip_R[1], FFTW_BACKWARD, FFTW_PATIENT);

            fftwf_export_wisdom_to_filename("phasecorr_fftw_wisdom.txt");

            // Populate fftshift matrices which are separable along each dimension
            fftshift = new double* [3];
            for (int n = 0; n < 3; n++)
            {
                fftshift[n] = new double[dims[n]];
                int idx = 0;
                for (int i = 0; i < dims[n] / 2; i++)
                {
                    fftshift[n][idx] = (double)i / (double)upsample_factor;
                    idx++;
                }
                for (int i = 0 - dims[n] / 2; i < 0; i++)
                {
                    fftshift[n][idx] = (double)i / (double)upsample_factor;
                    idx++;
                }
            }
        }

        inline void phaseCorr(double* dst)
        {
            for (int mip_idx = 2; mip_idx < 3; mip_idx++)  // For each MIP of the volume
            {

                // Calculate phase corr matrix R
                for (int i = 0; i < dims[0]; i++)
                {
                    for (int j = 0; j < dims[1]; j++)
                    {

                        // Convert to floats
                        memcpy(&i_tn, mip_tn[mip_idx] + i * dims[0] + j, sizeof(fftwf_complex));
                        memcpy(&i_t0, mip_t0[mip_idx] + i * dims[0] + j, sizeof(fftwf_complex));

                        // Correlation
                        pc = i_t0 * std::conj(i_tn);

                        if (std::abs(pc) == 0)
                        {
                            // Zero division case results in 0, not NaN
                            pcnorm = 0;
                        }
                        else
                        {
                            pcnorm = pc / std::abs(pc);
                        }

                        // Copy phase corr result to R array
                        memcpy(mip_r[mip_idx] + dims[0] * i + j, &pcnorm, sizeof(fftwf_complex));

                    }
                }

                // Convert R back to spatial domain
                fftwf_execute_dft(pc_mip_r_fft_plan, mip_r[mip_idx], mip_R[mip_idx]);

                maxval = -1;
                maxi = -1;
                maxj = -1;
                int r_norm = 0;

                // Naive search for max corr pixel
                int i;  // Will use later for centroid
                int j;
                for (i = 0; i < dims[0]; i++)
                {
                    for (j = 0; j < dims[1]; j++)
                    {
                        memcpy(&tmp, mip_R[mip_idx] + i * dims[0] + j, sizeof(fftwf_complex));
                        mag = std::abs(tmp);
                        r_norm += mag;

                        if (mag > maxval)
                        {
                            maxval = mag;
                            maxi = i;
                            maxj = j;
                        }
                    }
                }

                centroid_i = 0.0;
                centroid_j = 0.0;
                centroid_norm = 0.0;

                if (npeak > 0)
                {
                    for (int di = -npeak; di < npeak + 1; di++)
                    {
                        for (int dj = -npeak; dj < npeak + 1; dj++)
                        {
                            int shifti = mod(maxi + di, dims[0]);
                            int shiftj = mod(maxj + dj, dims[1]);
                            memcpy(&tmp, mip_R[mip_idx] + shifti * dims[0] + shiftj, sizeof(fftwf_complex));
                            centroid_norm += std::abs(tmp);
                            centroid_i += std::abs(tmp) * fftshift[mip_idx][shiftj];
                            centroid_j += std::abs(tmp) * fftshift[mip_idx][shifti];
                            // printf("Calculating centroid from point [%i, %i] = %f\n", mod(maxi + di, rwidth), mod(maxj + dj, rwidth), (float)std::abs(tmp));
                        }
                    }

                    maxval = maxval / centroid_norm;
                    dx = centroid_i / centroid_norm;
                    dy = centroid_j / centroid_norm;

                }
                else
                {
                    // maxval = maxval / acqWinWidth;
                    dx = fftshift[mip_idx][maxi];
                    dy = fftshift[mip_idx][maxj];
                }

                if (mip_idx == 2)
                {
                    memcpy(dst + 0, &maxval, sizeof(double));
                    memcpy(dst + 0, &dx, sizeof(double));
                    memcpy(dst + 1, &dy, sizeof(double));
                    memcpy(dst + 2, &dz, sizeof(double));
                }

            }  // For each B-scan

        }

        void setSpectralFilter(float* new_spectral_filter)
        {
            memcpy(spectral_filter, new_spectral_filter, dims[0] * dims[1] * sizeof(float));
        }

        void setSpatialFilter(float* new_spatial_filter)
        {
            memcpy(spatial_filter, new_spatial_filter, dims[0] * dims[1] * sizeof(float));
        }

        void setCentroidN(int n)
        {
            this->npeak = n;
        }

        void setBidirectional(bool is_bidirectional)
        {
            this->bidirectional = is_bidirectional;
        }

        void setReference(fftwf_complex* t0_new)
        {
            for (int cross_section = 0; cross_section < 3; cross_section++)
            {
                calculateMIP(mip_buffer, t0_new, cross_section);
                populateZeroPaddedBuffer2D(mip_t0[cross_section], mip_buffer);
                applyWindow2D(mip_t0[cross_section], spatial_filter);  // Multiply buffer by apod window prior to FFT
                fftwf_execute_dft(pc_mip_2d_fft_plan, mip_t0[cross_section], mip_t0[cross_section]);  // Execute in place FFT
                reference_acquired = true;
            }
        }

        void getDisplacement(fftwf_complex* frame, double* output)
        {
            if (reference_acquired) // Can only get displacement if a reference frame has been acquired
            {
                for (int cross_section = 2; cross_section < 3; cross_section++)
                {
                    calculateMIP(mip_buffer, frame, cross_section);
                    populateZeroPaddedBuffer2D(mip_tn[cross_section], mip_buffer);
                    applyWindow2D(mip_tn[cross_section], spatial_filter);  // Multiply buffer by apod window prior to FFT
                    fftwf_execute_dft(pc_mip_2d_fft_plan, mip_tn[cross_section], mip_tn[cross_section]);  // Execute in place FFT
                    reference_acquired = true;
                }
                phaseCorr(output);
            }
            else
            {
                printf("Cannot perform cross-correlation until a reference frame is acquired!\n");
            }
        }

        int get_frame_size()
        {
            return fsize;
        }

        // Return pointers to the buffers for display/debug

        fftwf_complex* get_R(int axis)
        {
            return mip_R[axis];
        }

        fftwf_complex* get_r(int axis)
        {
            return mip_r[axis];
        }

        fftwf_complex* get_tn(int axis)
        {
            return mip_tn[axis];
        }

        fftwf_complex* get_t0(int axis)
        {
            return mip_t0[axis];
        }
};