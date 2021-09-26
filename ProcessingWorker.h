#pragma once

#include <thread>
#include <atomic>
#include "fftw3.h"
#include <condition_variable>
#include <complex>
#include <chrono>
#include "SpscBoundedQueue.h"
#include "WavenumberInterpolationPlan.h"
#include "Utils.h"
#include <Windows.h>

enum ProcessingMessageFlag
{
	ProcessFrame = 1 << 1,
	ConfigureProcessor = 1 << 2,
};

DEFINE_ENUM_FLAG_OPERATORS(ProcessingMessageFlag)

struct ProcessingMessage
{
	ProcessingMessageFlag flag;
	std::atomic_int* ctr;  // Barrier counter to be incremented on completed task
	uint16_t* raw_src;  // Pointer to source of raw uint16 data
	float* bg_src;  // Pointer to average buffer which is subtracted from each A-line
	float* window_src;  // Window which is multiplied by each A-line
	fftwf_complex* dst_img;  // Destination for processed A-lines
	uint16_t* dst_stamp;  // Destination for A-line order stamp
	int aline_size;
	int number_of_alines;
	int roi_offset;
	int roi_size;
	WavenumberInterpolationPlan* interpdk_plan;
	fftwf_plan* fftw_plan;
};

typedef spsc_bounded_queue_t<ProcessingMessage> ProcessingQueue;

class ProcessingWorker final
{

protected:

	int id;

	std::thread processing_thread;

	ProcessingQueue* msg_queue;

	std::atomic_bool main_running;

	// Intermediate buffers
	float* spectral_buffer;
	fftwf_complex* spatial_buffer;

	int spatial_aline_size;

	// Used for processing
	float interp_dx;
	float interp_dy;
	float interp_y0;
	float interp_y1;

	inline void recv_msg()
	{
		ProcessingMessage msg;
		if (msg_queue->dequeue(msg))
		{
			if (msg.flag & ConfigureProcessor)
			{
				// Do any time-consuming prep for processing here

				// delete[] this->spectral_buffer;
				spectral_buffer = fftwf_alloc_real(msg.number_of_alines * msg.aline_size + 8 * msg.number_of_alines);
				spatial_aline_size = msg.aline_size / 2 + 1;
				spatial_buffer = fftwf_alloc_complex(msg.number_of_alines * spatial_aline_size);
			}
			if (msg.flag & ProcessFrame)
			{

				// k-linearization and DC subtraction
				for (int i = 0; i < msg.number_of_alines; i++)
				{
					// Copy and then zero the frame stamp
					msg.dst_stamp[i] = msg.raw_src[msg.aline_size * i];
					msg.raw_src[msg.aline_size * i] = 0;

					for (int j = 0; j < msg.aline_size; j++)  // For each element of each A-line
					{
						interp_y0 = msg.raw_src[msg.aline_size * i + msg.interpdk_plan->interp_map[0][j]] - msg.bg_src[msg.interpdk_plan->interp_map[0][j]];
						interp_y1 = msg.raw_src[msg.aline_size * i + msg.interpdk_plan->interp_map[1][j]] - msg.bg_src[msg.interpdk_plan->interp_map[1][j]];
						if (msg.interpdk_plan->interp_map[0][i] == msg.interpdk_plan->interp_map[1][i])
						{
							spectral_buffer[msg.aline_size * i + j] = interp_y0;
						}
						else
						{
							interp_dy = interp_y1 - interp_y0;
							interp_dx = msg.interpdk_plan->linear_in_k[j] - msg.interpdk_plan->linear_in_lambda[msg.interpdk_plan->interp_map[0][j]];
							spectral_buffer[msg.aline_size * i + j] = interp_y0 + interp_dx * (interp_dy / msg.interpdk_plan->d_lam);
						}
						// Multiply by window
						spectral_buffer[msg.aline_size * i + j] = spectral_buffer[msg.aline_size * i + j] * msg.window_src[j];
					}
				}
				// FFT into destination buffer
				fftwf_execute_dft_r2c(*(msg.fftw_plan), spectral_buffer, spatial_buffer);

				// Copy ROI to dest
				for (int i = 0; i < msg.number_of_alines; i++)
				{
					memcpy(msg.dst_img + i * msg.roi_size, spatial_buffer + i * spatial_aline_size + msg.roi_offset, msg.roi_size * sizeof(fftwf_complex));
				}

				// Increment the barrier counter
				(*msg.ctr)++;
			}
		}
	}

	void main()
	{
		while (main_running.load() == true)
		{
			this->recv_msg();
		}
	}

public:

	ProcessingWorker() {}

	ProcessingWorker(int thread_id)
	{
		id = thread_id;

		msg_queue = new ProcessingQueue(65536);
		processing_thread = std::thread(&ProcessingWorker::main, this);
		main_running = ATOMIC_VAR_INIT(true);
	}

	// DO NOT access non-atomics from outside main()

	void process_frame(std::atomic_int* ctr, int aline_size, int roi_offset, int roi_size, int number_of_alines, uint16_t* raw_src, float* bg_src, float* window_src,
	fftwf_complex* dst_img, uint16_t* dst_stamp, WavenumberInterpolationPlan* interpdk_plan, fftwf_plan* fftw_plan)
	{
		ProcessingMessage msg;
		msg.flag = ProcessFrame;
		msg.ctr = ctr;
		msg.aline_size = aline_size;
		msg.number_of_alines = number_of_alines;
		msg.roi_offset = roi_offset;
		msg.roi_size = roi_size;
		msg.raw_src = raw_src;
		msg.bg_src = bg_src;
		msg.window_src = window_src;
		msg.dst_img = dst_img;
		msg.dst_stamp = dst_stamp;
		msg.interpdk_plan = interpdk_plan;
		msg.fftw_plan = fftw_plan;
		msg_queue->enqueue(msg);
	}
	void configure(int aline_size, int number_of_alines)
	{
		ProcessingMessage msg;
		msg.flag = ConfigureProcessor;
		msg.aline_size = aline_size;
		msg.number_of_alines = number_of_alines;
		msg_queue->enqueue(msg);
	}

	void terminate()
	{
		main_running.store(false);
		processing_thread.join();
	}

	~ProcessingWorker()
	{
		fftwf_free(spectral_buffer);
		fftwf_free(spatial_buffer);
		// delete msg_queue;
	}

};
