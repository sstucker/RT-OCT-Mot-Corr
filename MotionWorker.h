#pragma once

#include <thread>
#include <atomic>
#include "fftw3.h"
#include <condition_variable>
#include <complex>
#include <chrono>
#include "SpscBoundedQueue.h"
#include "CircAcqBuffer.h"
#include "WavenumberInterpolationPlan.h"
#include "Utils.h"
#include <Windows.h>
#include "PhaseCorrelationPlan3D.h"
#include <NIDAQmx.h>

enum MotionMessageFlag
{
	Start = 1 << 1,
	Stop = 1 << 2,
	UpdateReference = 1 << 3,
	GrabCorrelogram = 1 << 4,
	GrabFrame = 1 << 5
};

DEFINE_ENUM_FLAG_OPERATORS(MotionMessageFlag);


struct MotionMessage
{
	MotionMessageFlag flag;
	CircAcqBuffer<fftwf_complex>* circacqbuffer;
	int spatial_aline_size;
	int upsample_factor;
	int* input_dims;
	int centroid_n_peak;  // 2 * centroid_n_peak + 1 is width of square ROI centered at correlogram max used to compute centroid
	float* window;
	fftwf_complex* grab_dst;  // Correlograms and frames are copied here for debugging and visualization
};

struct MotionVector
{
	double dx;
	double dy;
	double dz;
	int dt;  // The number of frames since reference frame t0. Unused
	double r;  // The complex-valued correlation of the frames. Unused
};

typedef spsc_bounded_queue_t<MotionMessage> MotionQueue;
typedef spsc_bounded_queue_t<MotionVector> MotionResultsQueue;

class MotionWorker final
{

protected:

	int id;

	std::thread mot_thread;

	MotionQueue* msg_queue;
	MotionResultsQueue* output_queue;
	TaskHandle motion_output_task;

	double* daq_xyz_out;
	int32* mot_samps_written;

	std::atomic_bool main_running;
	std::atomic_bool running;

	CircAcqBuffer<fftwf_complex>* acq_buffer;
	PhaseCorrelationPlan3D phase_correlation_plan;

	int buffer_size;

	bool update_reference;

	int openMotionOutputTask(const char* x_out_ch, const char* y_out_ch, const char* z_out_ch)
	{
		int err = DAQmxCreateTask("motion_output", &motion_output_task);

		if (err != 0)
		{
			printf("DAQmx error creating mot output task:\n");
			char* buf = new char[512];
			DAQmxGetErrorString(err, buf, 512);
			printf(buf);
			printf("\n");
			delete[] buf;
			return err;
		}

		err = DAQmxCreateAOVoltageChan(motion_output_task, x_out_ch, "x", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, y_out_ch, "y", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, z_out_ch, "z", -10, 10, DAQmx_Val_Volts, NULL);

		err = DAQmxCfgOutputBuffer(motion_output_task, 0);

		// err = DAQmxSetWriteRegenMode(motion_output_task, DAQmx_Val_DoNotAllowRegen);
		// err = DAQmxSetSampTimingType(motion_output_task, DAQmx_Val_OnDemand);
		// err = DAQmxSetSampTimingType(motion_output_task, DAQmx_Val_HWTimedSinglePoint);

		// err = DAQmxStartTask(motion_output_task);

		if (err != 0)
		{
			printf("DAQmx error configuring mot output task:\n");
			char* buf = new char[512];
			DAQmxGetErrorString(err, buf, 512);
			printf(buf);
			printf("\n");
			delete[] buf;
			return err;
		}

		daq_xyz_out = new double[3];  // Buffer for samples before they are written
		mot_samps_written = new int32[3];  // TODO multi-channel

		memset(daq_xyz_out, 0, 3 * sizeof(double));
		memset(mot_samps_written, 0, 3 * sizeof(int32));

		output_queue = new MotionResultsQueue(65536);

		return err;

	}

	inline void recv_msg()
	{
		MotionMessage msg;
		if (msg_queue->dequeue(msg))
		{
			if (msg.flag & Start)
			{
				if (!running.load())  // Ignore if already running
				{
					if (msg.input_dims[0] * msg.input_dims[1] * msg.input_dims[2] > 0)
					{
						
						if (openMotionOutputTask("Dev1/ao4", "Dev1/ao5", "Dev1/ao6") == 0)
						{
							acq_buffer = msg.circacqbuffer;
							fftwf_complex* mot_roi_buf = fftwf_alloc_complex(msg.input_dims[0] * msg.input_dims[1] * msg.input_dims[2]);
							buffer_size = (msg.input_dims[0] * msg.upsample_factor) * (msg.input_dims[1] * msg.upsample_factor) * (msg.input_dims[2] * msg.upsample_factor);
							phase_correlation_plan = PhaseCorrelationPlan3D(msg.input_dims, msg.upsample_factor, msg.centroid_n_peak, msg.window);
							running.store(true);
							update_reference = true;  // Always acquire a reference frame first

						}
						else
						{
							printf("Failed to open NI-DAQ motion output task\n");
						}
					}
				}
			}
			if (msg.flag & GrabCorrelogram)
			{
				memcpy(msg.grab_dst, phase_correlation_plan.get_R(), buffer_size * sizeof(fftwf_complex));
			}
			if (msg.flag & GrabFrame)
			{
				memcpy(msg.grab_dst, phase_correlation_plan.get_tn(), buffer_size * sizeof(fftwf_complex));
			}
			if (msg.flag & UpdateReference)
			{
				if (running.load())
				{
					update_reference = true;
				}
			}
			if (msg.flag & Stop)
			{
				if (running.load())
				{
					DAQmxStopTask(motion_output_task);
					DAQmxClearTask(motion_output_task);
					running.store(false);
				}
			}
		}
		else
		{
			return;
		}
	}

	void main()
	{
		printf("MotionWorker main() running...\n");

		bool fopen = false;
		int total = 0;
		int n_wanted = 0;
		int n_got;
		fftwf_complex* f;

		while (main_running.load() == true)
		{
			this->recv_msg();
			if (running.load())
			{
				n_got = acq_buffer->lock_out_wait(n_wanted, &f);
				// TODO if bidirectional, reorder

				if (n_wanted == n_got)
				{
					n_wanted += 1;
				}
				else
				{
					n_wanted = acq_buffer->get_count();
				}

				if (update_reference)
				{
					phase_correlation_plan.setReference(f);
					update_reference = false;  // Set flag back to 0
				}
				else
				{
					phase_correlation_plan.getDisplacement(f, daq_xyz_out);
					int err = DAQmxWriteAnalogF64(motion_output_task, 1, true, -1, DAQmx_Val_GroupByChannel, daq_xyz_out, mot_samps_written, NULL);
					if (err != 0)
					{
						printf("DAQmx error writing to DAC:\n");
						char* buf = new char[512];
						DAQmxGetErrorString(err, buf, 512);
						printf(buf);
						printf("\n");
						delete[] buf; 
					}

					MotionVector v = { daq_xyz_out[0], daq_xyz_out[1], daq_xyz_out[2] };
					output_queue->enqueue(v);

				}
				total += 1;
				acq_buffer->release();
			}
		}
	}

public:

	MotionWorker()
	{
		msg_queue = new MotionQueue(65536);
		main_running = ATOMIC_VAR_INIT(false);
		running = ATOMIC_VAR_INIT(false);
	}

	MotionWorker(int thread_id)
	{

		id = thread_id;

		msg_queue = new MotionQueue(65536);
		main_running = ATOMIC_VAR_INIT(true);
		running = ATOMIC_VAR_INIT(false);
		mot_thread = std::thread(&MotionWorker::main, this);
	}

	// DO NOT access anything thread unsafe from outside main()

	bool is_running()
	{
		return running.load();
	}

	void start(int spatial_aline_size, CircAcqBuffer<fftwf_complex>* buffer, int upsample_factor, int* input_dims, int centroid_n_peak, float* window)
	{
		MotionMessage msg;
		msg.flag = Start;
		msg.spatial_aline_size = spatial_aline_size;
		msg.circacqbuffer = buffer;
		msg.upsample_factor = upsample_factor;
		msg.input_dims = input_dims;
		msg.centroid_n_peak = centroid_n_peak;
		msg.window = window;
		msg_queue->enqueue(msg);
	}

	void grabCorrelogram(fftwf_complex* out)
	{
		MotionMessage msg;
		msg.flag = GrabCorrelogram;
		msg.grab_dst = out;
		msg_queue->enqueue(msg);
	}

	void grabFrame(fftwf_complex* out)
	{
		MotionMessage msg;
		msg.flag = GrabFrame;
		msg.grab_dst = out;
		msg_queue->enqueue(msg);
	}

	bool grabMotionVector(double* out)
	{
		MotionVector v;
		if (output_queue->dequeue(v))
		{
			out[0] = v.dx;
			out[1] = v.dy;
			out[2] = v.dz;
			return true;
		}
		else
		{
			return false;
		}
	}

	void stop()
	{
		MotionMessage msg;
		msg.flag = Stop;
		msg_queue->enqueue(msg);
	}

	void updateReference()
	{
		MotionMessage msg;
		msg.flag = UpdateReference;
		msg_queue->enqueue(msg);
	}

	void terminate()
	{
		main_running.store(false);
		mot_thread.join();
	}

};
