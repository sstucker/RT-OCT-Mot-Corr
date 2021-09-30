#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "fftw3.h"
#include "SpscBoundedQueue.h"
#include "ProcessingWorker.h"
#include "FileStreamWorker.h"
#include "MotionWorker.h"
#include "NIHardwareInterface.h"
#include "Utils.h"
#include "CircAcqBuffer.h"
#include <chrono>
#include <vector>
#include <ctime>
#include <Windows.h>


enum ControllerMessageFlag
{
	StopScan = 1 << 1,  // Stop the acquisition
	StartScan = 1 << 2,  // Start the acquisition
	UpdateScan = 1 << 3,  // Update the scan signals
	UpdateProcessing = 1 << 4,  // Update the processing parameters
	ConfigureController = 1 << 5  // Define hardware interfaces and allocate buffers (must not be scanning)
};

DEFINE_ENUM_FLAG_OPERATORS(ControllerMessageFlag)

// Data passed to the controller thread via MessageQueue
struct ControllerMessage
{
	ControllerMessageFlag flag;

	double* scan_x;
	double* scan_y;
	double* scan_line_trig;
	double* scan_frame_trig;
	int scan_n;
	int hw_dac_fs;
	int aline_size;
	int roi_offset;  // Offset of saved data from index 0 of each A-line
	int roi_size;  // Number of voxels saved of each A-line proceeding from offset
	int number_of_alines;
	int number_of_buffers;
	double intpdk;
	float* window;  // Spectrum apodization window
};

typedef spsc_bounded_queue_t<ControllerMessage> MessageQueue;

class RealtimeOCTController
{

protected:

	std::atomic_bool main_running;

	std::atomic_int worker_barrier;

	// TODO these probably don't need to be atomic
	std::atomic_bool config_ready;
	std::atomic_bool processing_ready;
	std::atomic_bool scan_ready;
	std::atomic_bool acquiring;

	NIHardwareInterface hardware_interface;
	uint16_t* raw_frame_addr;

	MessageQueue* msg_queue;
	std::thread oct_controller_thread;

	ProcessingWorker** processing_worker_pool;
	int n_process_workers;

	ProcessingMessage* worker_msgs;

	// Members read by threads
	CircAcqBuffer<fftwf_complex>* output_buffer;
	FileStreamWorker* file_stream_worker;

	// -- MOTION TRACKING -----------------------------------------------

	MotionWorker* motion_worker;

	// ------------------------------------------------------------------

	uint16_t* stamp_buffer;

	float* background_spectrum;
	float* background_spectrum_new;
	float* spectral_window;
	int aline_size;
	int spatial_aline_size;
	int total_number_of_alines;
	int alines_per_worker;
	double* scan_x;
	double* scan_y;
	double* scan_line_trig;
	double* scan_frame_trig;
	char* hw_cam_name;
	char* hw_ao_x;
	char* hw_ao_y;
	char* hw_ao_lt;
	char* hw_ao_ft;
	int roi_offset;
	int roi_size;

	bool save_stamp;

	fftwf_plan fftw_plan;
	WavenumberInterpolationPlan interpdk_plan;

	inline void launch_workers()  // Called from ScanStart
	{
		// TODO cleanup old workers

		processing_worker_pool = new ProcessingWorker * [n_process_workers];

		for (int i = 0; i < n_process_workers; i++)
		{
			processing_worker_pool[i] = new ProcessingWorker(i);
			processing_worker_pool[i]->configure(aline_size, alines_per_worker);
		}
	}
	 
	inline void terminate_workers()  // Called from ScanStop
	{
		for (int i = 0; i < n_process_workers; i++)
		{
			processing_worker_pool[i]->terminate();
		}
	}

	inline void plan_fftw()
	{
		// FFTW "many" plan
		int n[] = { aline_size };
		int idist = aline_size;
		int odist = spatial_aline_size;
		int istride = 1;
		int ostride = 1;
		int* inembed = n;
		int* onembed = &odist;
		float* dummy_in = fftwf_alloc_real(aline_size * alines_per_worker + 8 * alines_per_worker);
		fftwf_complex* dummy_out = fftwf_alloc_complex(spatial_aline_size * total_number_of_alines);

		auto start = std::clock();

		fftwf_import_wisdom_from_filename("C:/Users/OCT/Dev/RealtimeOCT/octcontroller_fftw_wisdom.txt");

		fftw_plan = fftwf_plan_many_dft_r2c(1, n, alines_per_worker, dummy_in, inembed, istride, idist, dummy_out, onembed, ostride, odist, FFTW_PATIENT);

		fftwf_export_wisdom_to_filename("C:/Users/OCT/Dev/RealtimeOCT/octcontroller_fftw_wisdom.txt");

		auto stop = std::clock();

		if (fftw_plan != NULL)
		{
			printf("Planned 32-bit %ix%i FFTW transform elapsed %f s\n", aline_size, alines_per_worker, std::difftime(stop, start));
		}
		else
		{
			printf("Failed to plan FFTW transform!\n");
		}
		printf("FFT plan created at %p\n", fftw_plan);
		fflush(stdout);

		fftwf_free(dummy_in);
		fftwf_free(dummy_out);
	}

	inline void recv_msg()  // Called from oct_controller_thread
	{
		ControllerMessage msg;
		if (msg_queue->dequeue(msg))
		{
			if (msg.flag & UpdateScan)
			{
				if (config_ready.load())
				{
					scan_ready.store(false);

					hardware_interface.set_scan_pattern(msg.scan_x, msg.scan_y, msg.scan_line_trig, msg.scan_frame_trig, msg.scan_n);

					scan_ready.store(true);
				}
				else
				{
					printf("Cannot update scan pattern: no scan task configured.\n");
				}
				
			}
			if (msg.flag & UpdateProcessing)
			{
				if (!file_stream_worker->is_streaming() && config_ready.load())
				{
					processing_ready.store(false);
					
					interpdk_plan = WavenumberInterpolationPlan(aline_size, msg.intpdk);
					spectral_window = new float[aline_size];
					memcpy(spectral_window, msg.window, aline_size * sizeof(float));

					processing_ready.store(true);
				}
				else
				{
					printf("Could not update processing. Controller is not configured, or file streaming is taking place.\n");
				}
			}
			if (msg.flag & ConfigureController)
			{
				if (!acquiring.load())
				{

					config_ready.store(false);

					aline_size = msg.aline_size;
					spatial_aline_size = aline_size / 2 + 1;
					total_number_of_alines = msg.number_of_alines;
					roi_offset = msg.roi_offset;
					roi_size = msg.roi_size;

					printf("Opening RealtimeOCTController with following hardware channels:\n");
					printf("Camera: %s\n", hw_cam_name);
					printf("X galvo: %s\n", hw_ao_x);
					printf("Y galvo: %s\n", hw_ao_y);
					printf("Line camera trigger: %s\n", hw_ao_lt);
					printf("Frame grab trigger: %s\n", hw_ao_ft);

					if (!hardware_interface.open(hw_cam_name, hw_ao_x, hw_ao_y, hw_ao_lt, hw_ao_ft, msg.hw_dac_fs, aline_size, total_number_of_alines, msg.number_of_buffers))
					{
						// Determine number of workers based on A-lines per frame
						if (total_number_of_alines > 4096)
						{
							n_process_workers = std::thread::hardware_concurrency();
						}
						else
						{
							n_process_workers = std::thread::hardware_concurrency() / 2;
						}

						while ((total_number_of_alines % n_process_workers != 0) && (n_process_workers > 1))
						{
							n_process_workers -= 1;
						}

						alines_per_worker = total_number_of_alines / n_process_workers;

						save_stamp = false;  // TODO parameter

						// Allocate background spectrum
						background_spectrum = new float[aline_size];
						background_spectrum_new = new float[aline_size];
						memset(background_spectrum, 0, aline_size * sizeof(float));
						memset(background_spectrum_new, 0, aline_size * sizeof(float));

						// TODO free old output buffer if it exists

						output_buffer = new CircAcqBuffer<fftwf_complex>(msg.number_of_buffers, roi_size * total_number_of_alines);
						stamp_buffer = new uint16_t[total_number_of_alines];

						plan_fftw();

						config_ready.store(true);
					}
					else
					{
						printf("Config failed. Cannot open hardware interface!\n");
					}
					
				}
				else
				{
					printf("Cannot configure during acquisition!\n");
				}
			}
			if (msg.flag & StartScan)
			{
				if (config_ready.load() && processing_ready.load() && scan_ready.load())
				{
					
					launch_workers();
					
					if (!hardware_interface.start_scan())
					{
						printf("Hardware started scanning successfully.\n");
						acquiring.store(true);
					}
					else
					{
						printf("Hardware failed to start scan!\n");
					}
				}
				else
				{
					printf("Controller not ready to scan.\n");
				}
			}
			if (msg.flag & StopScan)
			{
				if (acquiring.load())
				{
					if (!hardware_interface.stop_scan())
					{
						terminate_workers();
						acquiring.store(false);
						fflush(stdout);
					}
				}
				else
				{
					printf("Can't stop... not scanning!\n");
				}
			}
		}
	}

	void main()  // oct_controller_thread
	{

		int cumulative_frame_number = 0;
		int acquired_buffer_number = 0;

		fftwf_complex* output_addr = NULL;

		while (main_running.load() == true)
		{
			// Poll msg buffer
			recv_msg();

			if (acquiring.load())
			{
				acquired_buffer_number = hardware_interface.examine_imaq_buffer(&raw_frame_addr, cumulative_frame_number);
				
				if ((raw_frame_addr != NULL) && (acquired_buffer_number > -1))
				{					    
					// printf("Acq buf num %i\n", acquired_buffer_number);
					output_addr = output_buffer->lock_out_head();

					// Dispatch jobs to workers
					worker_barrier.store(0);
					for (int i = 0; i < n_process_workers; i++)
					{
						processing_worker_pool[i]->process_frame(&worker_barrier, aline_size, roi_offset, roi_size, alines_per_worker,
							raw_frame_addr + (alines_per_worker * aline_size * i), background_spectrum, spectral_window,
							output_addr + (alines_per_worker * roi_size * i), stamp_buffer + (alines_per_worker * i),
							&interpdk_plan, &fftw_plan);
					}

					// Compute background spectrum to be used with next frame while workers compute
					memset(background_spectrum_new, 0, aline_size * sizeof(float));
					
					for (int i = 0; i < total_number_of_alines; i++)
					{
						for (int j = 0; j < aline_size; j++)
						{
							background_spectrum_new[j] += (float)raw_frame_addr[i * aline_size + j] / (float)total_number_of_alines;
						}
					}

					// Spinlock on barrier TODO add timeout
					while (worker_barrier.load() < n_process_workers);

					// Swap in new background buffer once workers finish with old one
					float* tmp = background_spectrum;
					background_spectrum = background_spectrum_new;
					background_spectrum_new = tmp;

					// Assign line camera stamp real value of first voxel
					if (save_stamp)
					{
						for (int i = 0; i < total_number_of_alines; i++)
						{
							output_addr[roi_size * i][0] = stamp_buffer[i];
						}
					}

					output_buffer->release_head();

					cumulative_frame_number += 1;

				}
				hardware_interface.release_imaq_buffer();
			}
		}

		hardware_interface.close();

		delete output_buffer;
		delete[] stamp_buffer;
		delete[] background_spectrum;

		fflush(stdout);

	}

public:

	// Do not access properties here

	RealtimeOCTController(RealtimeOCTController&&) {}  // TODO implement this

	RealtimeOCTController()
	{
		config_ready = ATOMIC_VAR_INIT(false);
		processing_ready = ATOMIC_VAR_INIT(false);
		scan_ready = ATOMIC_VAR_INIT(false);
		acquiring  = ATOMIC_VAR_INIT(false);
		main_running = ATOMIC_VAR_INIT(false);

		// TODO initialize properties
	}

	bool is_scanning()
	{
		return acquiring.load();
	}

	bool is_ready_to_scan()
	{
		return config_ready.load() && processing_ready.load() && scan_ready.load();
	}

	// Starts all threads
	void open(const char* cam_name, const char* aoScanX, const char* aoScanY, const char* aoLineTrigger, const char* aoFrameTrigger)
	{
		if (!main_running.load())
		{
			hw_cam_name = new char[strlen(cam_name) + 1];
			strcpy(hw_cam_name, cam_name);

			hw_ao_x = new char[strlen(aoScanX) + 1];
			strcpy(hw_ao_x, aoScanX);

			hw_ao_y = new char[strlen(aoScanY) + 1];
			strcpy(hw_ao_y, aoScanY);

			hw_ao_lt = new char[strlen(aoLineTrigger) + 1];
			strcpy(hw_ao_lt, aoLineTrigger);

			hw_ao_ft = new char[strlen(aoFrameTrigger) + 1];
			strcpy(hw_ao_ft, aoFrameTrigger);

			msg_queue = new MessageQueue(65536);

			oct_controller_thread = std::thread(&RealtimeOCTController::main, this);
			file_stream_worker = new FileStreamWorker(0);  // If no argument is passed, thread does not start
			motion_worker = new MotionWorker(0);

			main_running.store(true);
		}
	}

	int configure(int dac_rate, int aline_size, int roi_offset, int roi_size, int number_of_alines, int number_of_buffers)
	{
		ControllerMessage msg;
		msg.flag = ConfigureController;
		msg.hw_dac_fs = dac_rate;
		msg.aline_size = aline_size;
		msg.roi_offset = roi_offset;
		msg.roi_size = roi_size;
		msg.number_of_alines = number_of_alines;
		msg.number_of_buffers = number_of_buffers;

		msg_queue->enqueue(msg);

		return 0;
	}

	void set_oct_processing(double intpdk, float* window)
	{
		ControllerMessage msg;
		msg.flag = UpdateProcessing;
		msg.intpdk = intpdk;
		msg.window = window;
		msg_queue->enqueue(msg);
	}

	void set_scan_signals(double* x, double* y, double* linetrigger, double* frametrigger, int n)
	{
		ControllerMessage msg;
		msg.flag = UpdateScan;
		msg.scan_x = x;
		msg.scan_y = y;
		msg.scan_line_trig = linetrigger;
		msg.scan_frame_trig = frametrigger; 
		msg.scan_n = n;
		msg_queue->enqueue(msg);
	}

	void start_scan()
	{
		ControllerMessage msg;
		msg.flag = StartScan;
		msg_queue->enqueue(msg);
	}

	void stop_scan()
	{
		ControllerMessage msg;
		msg.flag = StopScan;
		msg_queue->enqueue(msg);
	}

	// Grabs the average spectrum of the most recently acquired frame, copying it into an output buffer
	void grab_spectrum(float* out)
	{
		memcpy(out, background_spectrum, aline_size * sizeof(float));
	}

	// Grabs the most recently acquired frame, copying it into an output buffer, and returning its cumulative index
	int grab_frame(fftwf_complex* out)
	{
		if (acquiring.load())
		{
			int want = output_buffer->get_count();  // Returns latest frame
			if (want > -1)
			{
				memcpy(out, (*output_buffer)[want], total_number_of_alines * roi_size * sizeof(fftwf_complex));
				return want;  // Return index of grabbed frame
			}
			else
			{
				return -1;
			}
		}
		else
		{
			return -1;
		}
	}

	void start_save(const char* fname, int max_bytes)
	{
		// while (!acquiring.load())
		file_stream_worker->start_streaming(fname, max_bytes, FSTREAM_TYPE_NPY, output_buffer, roi_size, total_number_of_alines, roi_size);
	}

	void save_n(const char* fname, int max_bytes, int n_to_save)
	{
		if (acquiring.load())
		{
			file_stream_worker->start_streaming(fname, max_bytes, FSTREAM_TYPE_NPY, output_buffer, roi_size, total_number_of_alines, n_to_save);
		}
	}

	void stop_save()
	{
		file_stream_worker->stop_streaming();
	}

	// -- MOTION QUANT ---------------------

	void start_motion_output(int* input_dims, double* scale_xyz, int upsample_factor, int centroid_n_peak, float* spectral_filter, float* spatial_filter, bool bidirectional,
		                     double* filter_d, double* filter_g, double* filter_q, double* filter_r)
	{
		motion_worker->start(spatial_aline_size, output_buffer, upsample_factor, input_dims, scale_xyz, centroid_n_peak, spectral_filter, spatial_filter, bidirectional, filter_d, filter_g, filter_q, filter_r);
	}

	void stop_motion_output()
	{
		motion_worker->stop();
	}

	void update_motion_reference()
	{
		motion_worker->updateReference();
	}

	void update_motion_parameters(double* scale_xyz, int centroid_n_peak, float* spectral_filter, float* spatial_filter, bool bidirectional, double* filter_d, double* filter_g, double* filter_q, double* filter_r)
	{
		motion_worker->updateParameters(scale_xyz, centroid_n_peak, spectral_filter, spatial_filter, bidirectional, filter_d, filter_g, filter_q, filter_r);
	}

	void grab_motion_correlogram(fftwf_complex* out)
	{
		motion_worker->grabCorrelogram(out);
	}

	void grab_motion_frame(fftwf_complex* out)
	{
		motion_worker->grabFrame(out);
	}

	int grab_motion_vector(double* out)
	{
		return motion_worker->grabMotionVector(out);
	}

	// ----------------

	void close()
	{
		main_running.store(false);
		printf("RealtimeOCTController closing...\n");
		oct_controller_thread.join();
		file_stream_worker->terminate();  // Joins fstream
	}

	~RealtimeOCTController()
	{
		printf("RealtimeOCTController destructor invoked\n");
		delete[] hw_cam_name;
		delete[] hw_ao_x;
		delete[] hw_ao_y;
		delete[] hw_ao_lt;
		delete[] hw_ao_ft;
		delete file_stream_worker;
		/*
		for (int i = 0; i < n_process_workers; i++)
		{
			delete processing_worker_pool[i];
		}
		delete[] processing_worker_pool;
		*/
	}

};
