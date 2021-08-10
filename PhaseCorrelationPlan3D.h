#pragma once
#include "fftw3.h"
#include <complex>
#include "Utils.h"



class PhaseCorrelationPlan3D
{

	protected:
		int* dims;
        int* og_dims;
		int fsize;
        int og_fsize;

        bool reference_acquired;  // true if a reference frame exists

        fftwf_complex* t0;  // Reference frame ROI for phase corr
        fftwf_complex* tn;  // Delta frame ROI for phase corr
        fftwf_complex* r; // Phase correlation output
        fftwf_complex* R;  // Spatial correlogram result of inv FFT

        // Buffers for correlation calc
        std::complex<float> i_t0;
        std::complex<float> i_tn;
        std::complex<float> pc;
        std::complex<float> pcnorm;

        int* px_dx_lut;  // LUT to convert from FFT pixels to pixel displacements for a square ROI
        int pc_roi;  // The offset into the top of the A-scan from which to crop the bwidth x bwidth ROI

        float* apod_window; // Pixmap of attenuation values multiplied by the 2D spatial images prior to forward FFT 

        // 32 bit 2D fft plans for phase correlation
        fftwf_plan pc_roi_fft_plan;
        fftwf_plan pc_r_fft_plan;

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
        float* avg_corr_x;
        float* avg_corr_y;
        float* avg_pos_x;
        float* avg_pos_y;
        float maxval;
        float mag;
        std::complex<float> tmp;
        double** fftshift;

        int var_idx;
        int var_window_size;
        float** var_buffers;
        float dot_thresh;
        float var_b;

        template <class T>
        inline T* indexBuffer(T* buf, int i, int j, int k)
        {
            // printf("Indexing at [%i][%i][%i] -> %i\n", i, j, k, k + dims[2] * j + dims[2] * dims[1] * i);
            return buf + (i + dims[1] * (j + dims[2] * k));
        }

        inline void applyWindow(fftwf_complex* img, float* window)
        {
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    for (int k = 0; k < dims[2]; k++)
                    {
                        // Complex number
                        *(indexBuffer<fftwf_complex>(img, i, j, k))[0] *= *indexBuffer<float>(window, i, j, k);
                        *(indexBuffer<fftwf_complex>(img, i, j, k))[1] *= *indexBuffer<float>(window, i, j, k);
                    }
                }
            }
        }

        inline void populateZeroPaddedBuffer(fftwf_complex* dst, fftwf_complex* src)
        {
            // Copy from src buffer with dimensions i, j, k ... into center of zero-padded buffer
            // of size ni nj nk ... for upsampling

            memset(dst, 0, fsize * sizeof(fftwf_complex));  // Set the destination to 0, as it is zero-padded

            for (int i = 0; i < og_dims[0]; i++)
            {
                for (int j = 0; j < og_dims[1]; j++)
                {
                    for (int k = 0; k < og_dims[2]; k++)
                    {
                        int dst_i = (dims[0] - og_dims[0]) / 2 + i;
                        int dst_j = (dims[1] - og_dims[1]) / 2 + j;
                        int dst_k = (dims[2] - og_dims[2]) / 2 + k;
                        // TODO memcpy last axis contiguously
                        memcpy(indexBuffer<fftwf_complex>(dst, dst_i, dst_j, dst_k), src + (i + og_dims[1] * (j + og_dims[2] * k)), sizeof(fftwf_complex));
                    }
                }
            }
        }

        inline void phaseCorr(double* dst)
        {
            // Calculate phase corr matrix R
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    for (int k = 0; k < dims[2]; k++)
                    {
                        // Convert to floats
                        memcpy(&i_tn, indexBuffer<fftwf_complex>(tn, i, j, k), sizeof(fftwf_complex));
                        memcpy(&i_t0, indexBuffer<fftwf_complex>(t0, i, j, k), sizeof(fftwf_complex));

                        // printf("tn[%i][%i][%i] = %f + %fi\n", i, j, k, *(indexBuffer<fftwf_complex>(tn, i, j, k))[0], *(indexBuffer<fftwf_complex>(tn, i, j, k))[1]);
                        // printf("t0[%i][%i][%i] = %f + %fi\n", i, j, k, *(indexBuffer<fftwf_complex>(t0, i, j, k))[0], *(indexBuffer<fftwf_complex>(t0, i, j, k))[1]);

                        // printf("tn[%i][%i][%i] = %f + %fi\n", i, j, k, std::real(i_tn), std::imag(i_tn));
                        // printf("t0[%i][%i][%i] = %f + %fi\n", i, j, k, std::real(i_t0), std::imag(i_t0));

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
                        memcpy(indexBuffer<fftwf_complex>(r, i, j, k), &pcnorm, sizeof(fftwf_complex));
                    }
                }
            }

            // Convert R back to spatial domain
            fftwf_execute_dft(pc_r_fft_plan, r, R);

            maxval = -1;
            maxi = -1;
            maxj = -1;
            maxk = -1;
            int r_norm = 0;

            // Naive search for max corr pixel
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    for (int k = 0; k < dims[2]; k++)
                    {
                        memcpy(&tmp, indexBuffer<fftwf_complex>(R, i, j, k), sizeof(fftwf_complex));
                        mag = std::abs(tmp);
                        // printf("Value of correlogram %f + %fi\n", *(indexBuffer<fftwf_complex>(R, i, j, k))[0], *(indexBuffer<fftwf_complex>(R, i, j, k))[1]);
                        r_norm += mag;

                        if (mag > maxval)
                        {
                            maxval = mag;
                            maxi = i;
                            maxj = j;
                            maxk = k;
                            // printf("New max = %f at [%i][%i][%i]\n", mag, i, j, k);
                        }
                    }
                }
            }
            // printf("Correlogram max at [%i, %i, %i]\n", maxi, maxj, maxk);

            centroid_i = 0.0;
            centroid_j = 0.0;
            centroid_k = 0.0;
            centroid_norm = 0.0;

            if (npeak > 0)
            {
                for (int di = -npeak; di < npeak + 1; di++)
                {
                    for (int dj = -npeak; dj < npeak + 1; dj++)
                    {
                        for (int dk = -npeak; dk < npeak + 1; dk++)
                        {
                            int shifti = mod(maxi + di, dims[0]);
                            int shiftj = mod(maxj + dj, dims[1]);
                            int shiftk = mod(maxk + dk, dims[2]);
                            memcpy(&tmp, indexBuffer<fftwf_complex>(R, shifti, shiftj, shiftk), sizeof(fftwf_complex));
                            // printf("tmp = %f at [%i][%i][%i]\n", std::abs(tmp), shifti, shiftj, shiftk);
                            centroid_norm += std::abs(tmp);
                            // printf("fftshift = %f at [%i]\n", fftshift[0][shifti], shifti);
                            centroid_i += std::abs(tmp) * fftshift[0][shifti];
                            centroid_j += std::abs(tmp) * fftshift[1][shiftj];
                            centroid_k += std::abs(tmp) * fftshift[2][shiftk];
                        }
                    }
                }

                maxval = maxval / centroid_norm;
                // printf("dx = %f / %f\n", centroid_i, centroid_norm);
                dx = centroid_i / centroid_norm;
                dy = centroid_j / centroid_norm;
                dz = centroid_k / centroid_norm;

            }
            else
            {
                dx = fftshift[0][maxi];
                dy = fftshift[1][maxj];
                dz = fftshift[2][maxj];
            }

            memcpy(dst + 0, &maxval, sizeof(double));
            memcpy(dst + 0, &dx, sizeof(double));
            memcpy(dst + 1, &dy, sizeof(double));
            memcpy(dst + 2, &dy, sizeof(double));

        }


	public:

		PhaseCorrelationPlan3D()
        {
            reference_acquired = false;
        }

        PhaseCorrelationPlan3D(int* input_dims, int upsample, int npeak_centroid, float* window)
        {

            reference_acquired = false;

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

            printf("Planning 3D phase correlation:\n");
            printf("Upsampling [%i, %i, %i] -> [%i, %i, %i]\n", og_dims[0], og_dims[1], og_dims[2], dims[0], dims[1], dims[2]);
            printf("Allocating buffers of size %i\n", fsize);
            apod_window = new float[fsize];
            memset(apod_window, 1.0, fsize * sizeof(float));
            memcpy(apod_window, window, fsize * sizeof(float));

            t0 = fftwf_alloc_complex(fsize);
            tn = fftwf_alloc_complex(fsize);
            r = fftwf_alloc_complex(fsize);
            R = fftwf_alloc_complex(fsize);

            fftwf_import_wisdom_from_filename("phasecorr_fftw_wisdom.txt");

            printf("Planning FFTWF transform...\n");
            pc_roi_fft_plan = fftwf_plan_dft(3, dims, t0, t0, FFTW_FORWARD, FFTW_PATIENT);
            pc_r_fft_plan = fftwf_plan_dft(3, dims, r, R, FFTW_BACKWARD, FFTW_PATIENT);

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

            
            printf("fftshift matrix for upsample factor %i:\n", upsample_factor);
            for (int i = 0; i < dims[0]; i++)
            {
                printf("%f ", fftshift[0][i]);
            }
            printf("\n");
            
        }


        void setReference(fftwf_complex* t0_new)
        {
            populateZeroPaddedBuffer(t0, t0_new);
            reference_acquired = true;
            applyWindow(t0, apod_window);  // Multiply buffer by apod window prior to FFT
            fftwf_execute_dft(pc_roi_fft_plan, t0, t0);  // Execute in place FFT
            // TODO consider filtering the spectrum
            printf("Got new reference frame.\n");
        }

        void getDisplacement(fftwf_complex* frame, double* output)
        {
            if (reference_acquired)
            {
                populateZeroPaddedBuffer(tn, frame);
                applyWindow(tn, apod_window);  // Multiply buffer by apod window prior to FFT
                fftwf_execute_dft(pc_roi_fft_plan, tn, tn);  // Execute in place FFT
                phaseCorr(output);
            }
            else
            {
                printf("Cannot perform cross-correlation until a reference frame is acquired!\n");
            }
        }

        fftwf_complex* get_R()
        {
            return R;
        }

        fftwf_complex* get_tn()
        {
            return tn;
        }

        fftwf_complex* get_t0()
        {
            return t0;
        }

		~PhaseCorrelationPlan3D()
		{
            // delete[] dims;
            // delete[] og_dims;
            // delete[] apod_window;
		}



};