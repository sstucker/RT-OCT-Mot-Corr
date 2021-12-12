#pragma once

#include <thread>
#include <atomic>
#include "fftw3.h"
#include <condition_variable>
#include <algorithm>
#include <complex>
#include <chrono>
#include "SpscBoundedQueue.h"
#include "CircAcqBuffer.h"
#include "WavenumberInterpolationPlan.h"
#include "Utils.h"
#include <Windows.h>
#include "PhaseCorrelationPlan3D.h"
#include "PhaseCorrelationPlanMIP3.h"
#include <NIDAQmx.h>
#include "SimpleKalmanFilter.h"
#include <Eigen/Dense>

enum MotionMessageFlag
{
	Start = 1 << 1,
	Stop = 1 << 2,
	UpdateReference = 1 << 3,
	GrabCorrelogram = 1 << 4,
	GrabFrame = 1 << 5,
	UpdateParameters = 1 << 6,
	ConfigOutput = 1 << 7,
	RunExperiment = 1 << 8
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
	float* spatial_filter;
	float* spectral_filter;
	fftwf_complex* grab_dst;  // Correlograms and frames are copied here for debugging and visualization
	bool bidirectional;  // If true, the voxels of every other B-scan (2nd axis) is reversed prior to correlation TODO
	bool kf_type;  // TODO
	float pattern_period;  // Number of seconds between each frame for velocity calculation
	double* filter_e;  // Proportion of decay of position to 0 between time steps. Value of 1 -> no decay
	double* filter_f;  // Proportion of decay of velocity to 0 between time steps
	double* filter_g;  // Proportion of decay of acceleration to 0 between time steps
	double* filter_q;  // Kalman process noise covariance diag value
	double* filter_r1;  // Kalman position measurement noise covariance value
	double* filter_r2;  // Kalman velocity measurement noise covariance value
	double filter_dt;  // Kalman velocity measurement noise covariance value
	int n_lag;  // Difference of frames to use for calculating velocity
	double* scale_xyz;  // Scale factor per x, y, z such that DAC output corresponds to desired spatial units. Default 1/4
	bool output_enabled; // Whether or not output is written to DAC
	const char* ao_dx_ch;  // DAC correction signal generation channels
	const char* ao_dy_ch;
	const char* ao_dz_ch;
	const char* ao_trig_ch;
	int exp_n_stim;  // Experiment: number of aux2 pulses to generate
	int exp_wait_seconds;  // Experiment: number of passive seconds to record before and after each aux2 pulse
};

struct MotionVector
{
	double x;
	double y;
	double z;
	double dx;
	double dy;
	double dz;
	double filtered_x;
	double filtered_y;
	double filtered_z;
	double aux1;  // Aux channel (Experiment start trigger)
	double aux2;  // Aux channel (Air puff stimulus trigger)
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

	bool dac_output_enabled;
	double* correlation_out;
	double* daq_xyz_out;
	double* daq_xyz_scale;
	SimpleKalmanFilter* filters_xyz;
	Eigen::VectorXd* filter_input_xyz;
	int32* mot_samps_written;

	std::atomic_bool main_running;
	std::atomic_bool running;

	CircAcqBuffer<fftwf_complex>* acq_buffer;
	
	PhaseCorrelationPlan3D phase_correlation_plan;

	int buffer_size;
	int frame_size;

	bool update_reference;
	bool filters_enabled;
	
	bool experiment_running;
	bool experiment_start;
	int exp_n_stim; // Number of stimuli left to acquire
	int exp_wait_sec;  // Seconds to wait before or after each stimuli

	int initializeCAFilters(double* d, double* f, double* g, double* q, double* r1, double* r2, double dt)
	{
		int n = 3; // x, dx, ddx
		int m = 2; // Measurements on x and dx

		// Construct a separate filter for each dimension, x, y and z
		for (int i = 0; i < 3; i++)
		{
			Eigen::MatrixXd A(n, n); // System dynamics matrix
			Eigen::MatrixXd H(m, n); // Measurement matrix
			Eigen::MatrixXd Q(n, n); // Process noise covariance
			Eigen::MatrixXd R(m, m); // Measurement noise covariance
			Eigen::MatrixXd P0(n, n); // Estimate error covariance
			Eigen::VectorXd X0(n);  // Init state

			// State transition model
			A << d[i], dt,   dt*dt / 2,
				 0,    f[i], dt,
				 0,    0,    g[i];

			// Measurement matrix (Measuring on position and velocity)
			H << 1, 0, 0,
			   	 0, 1, 0;


			// Process noise covariance
			Q << 0, 0, 0,
				 0, 0, 0,
				 0, 0, q[i];

			// Measurement covariance (Measuring on position)
			R << r1[i], 0,
				 0,     r2[i];

			// Initial P
			P0 << 0, 0, 0,
				  0, 0, 0,
				  0, 0, 0;

			// Initial state
			X0 << 0, 0, 0;

			filters_xyz[i] = SimpleKalmanFilter(A, H, Q, R, X0, P0);
			filter_input_xyz[i] = Eigen::VectorXd(2);
		}
		return 0;
	}

	int openMotionOutputTask(const char* x_out_ch, const char* y_out_ch, const char* z_out_ch, const char* aux1_ch, const char* aux2_ch)
	{
		if (running.load())
		{
			DAQmxStopTask(motion_output_task);
			DAQmxClearTask(motion_output_task);
		}

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

		printf("Opening motion output task with AO channels %s %s %s %s %s\n", x_out_ch, y_out_ch, z_out_ch, aux1_ch, aux2_ch);

		err = DAQmxCreateAOVoltageChan(motion_output_task, x_out_ch, "x", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, y_out_ch, "y", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, z_out_ch, "z", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, aux1_ch, "aux1", -10, 10, DAQmx_Val_Volts, NULL);
		err = DAQmxCreateAOVoltageChan(motion_output_task, aux2_ch, "aux2", -10, 10, DAQmx_Val_Volts, NULL);

		// err = DAQmxCfgSampClkTiming(motion_output_task, NULL, 200, DAQmx_Val_Rising, DAQmx_Val_OnDemand, 1);
		// err = DAQmxCfgOutputBuffer(motion_output_task, 0);

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

		printf("Motion output task configured successfully.\n");

		return err;

	}

	inline void recv_msg()
	{
		MotionMessage msg;
		if (msg_queue->dequeue(msg))
		{
			if (msg.flag & ConfigOutput)
			{
				// TODO sort out the channel input strings
				if (openMotionOutputTask("Dev1/ao4", "Dev1/ao5", "Dev1/ao6", "Dev1/ao7", "Dev1/ao8") == 0)
				{
					dac_output_enabled = msg.output_enabled;
					memcpy(daq_xyz_scale, msg.scale_xyz, 3 * sizeof(double));
				}
			}
			if (msg.flag & Start)
			{
				if (!running.load())  // Ignore if already running
				{
					if (msg.input_dims[0] * msg.input_dims[1] * msg.input_dims[2] > 0)
					{
						acq_buffer = msg.circacqbuffer;
						fftwf_complex* mot_roi_buf = fftwf_alloc_complex(msg.input_dims[0] * msg.input_dims[1] * msg.input_dims[2]);
						buffer_size = (msg.input_dims[0] * msg.upsample_factor) * (msg.input_dims[1] * msg.upsample_factor) * (msg.input_dims[2] * msg.upsample_factor);
						frame_size = msg.input_dims[0] * msg.input_dims[1] * msg.input_dims[2];
							
						phase_correlation_plan = PhaseCorrelationPlan3D(msg.input_dims, msg.upsample_factor, msg.centroid_n_peak, msg.spectral_filter, msg.spatial_filter, msg.bidirectional, msg.n_lag);

						initializeCAFilters(msg.filter_e, msg.filter_f, msg.filter_g, msg.filter_q, msg.filter_r1, msg.filter_r2, msg.filter_dt);  // TODO dt based on pattern rate

						filters_enabled = true;  // todo parameter

						running.store(true);
						update_reference = true;  // Always acquire a reference frame first
					}
				}
			}
			if (msg.flag & UpdateParameters)
			{
				phase_correlation_plan.setSpectralFilter(msg.spectral_filter);
				phase_correlation_plan.setSpatialFilter(msg.spatial_filter);
				phase_correlation_plan.setCentroidN(msg.centroid_n_peak);
				phase_correlation_plan.setBidirectional(msg.bidirectional);
				initializeCAFilters(msg.filter_e, msg.filter_f, msg.filter_g, msg.filter_q, msg.filter_r1, msg.filter_r2, msg.filter_dt);
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
			if (msg.flag & RunExperiment)
			{
				if (running.load() && !experiment_running)  // Only start an experiment if one is not ongoing
				{
					exp_n_stim = msg.exp_n_stim;
					exp_wait_sec = msg.exp_wait_seconds;
					printf("Running experiment with %i stims\n", exp_n_stim);
					experiment_running = true;
					experiment_start = true;
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
			// printf("No message!\n");
		}
	}

	void main()
	{
		printf("MotionWorker main() running...\n");

		exp_n_stim = 0;
		exp_wait_sec = 0;
		std::chrono::steady_clock::time_point exp_t_last_stim;
		std::chrono::steady_clock::time_point exp_t_start;

		experiment_running = false;
		experiment_start = false;

		bool fopen = false;
		int total = 0;
		int n_got;
		int n_wanted = 0;
		fftwf_complex* f;

		while (main_running.load() == true)
		{
			recv_msg();

			if (running.load())
			{

				if (update_reference)
				{
					int averaged = 0;
					int to_average = 10;  // TODO argument of update reference
					fftwf_complex* average_frame = fftwf_alloc_complex(frame_size);
					memset(average_frame, 0, sizeof(fftwf_complex) * frame_size);
					while (averaged < to_average)  // Calculate average
					{
						n_got = acq_buffer->lock_out_wait(acq_buffer->get_count(), &f);
						if (n_got > -1)
						{
							for (int i = 0; i < frame_size; i++)
							{
								average_frame[i][0] += f[i][0] / to_average;
								average_frame[i][1] += f[i][1] / to_average;
							}
							averaged += 1;
						}
						acq_buffer->release();
					}
					phase_correlation_plan.setReference(average_frame);
					printf("Updated reference\n");
					update_reference = false;  // Set flag back to 0
					fftwf_free(average_frame);
				}

				n_got = acq_buffer->lock_out_wait(n_wanted, &f);
				if (n_wanted != n_got)
				{
					n_wanted = acq_buffer->get_count();
				}

				// TODO if bidirectional, reorder
				if (n_got > -1)
				{
					phase_correlation_plan.getTotalAndIncrementalDisplacement(f, correlation_out);
					// printf("correlation_out = [%f, %f, %f, %f, %f, %f]\n", correlation_out[0], correlation_out[1], correlation_out[2], correlation_out[3], correlation_out[4], correlation_out[5]);

					acq_buffer->release();

					for (int i = 0; i < 3; i++)  // For x, y, z
					{
						if (filters_enabled)
						{
							filter_input_xyz[i] << correlation_out[i], correlation_out[3 + i];  // Load algorithm output into Eigen matrix
							filters_xyz[i].observeAndPredict(filter_input_xyz[i]);  // Kalman update and predict
							daq_xyz_out[i] = filters_xyz[i].getState()[0];  // Get first state var, position
						}
						daq_xyz_out[i] = daq_xyz_out[i] * daq_xyz_scale[i];
					} 

					/*
					// Testing clock
					daq_xyz_out[0] = (total % 2 == 0);
					daq_xyz_out[1] = (total % 2 == 0);
					daq_xyz_out[2] = (total % 2 == 0);
					*/

					if (experiment_running)
					{
						if (experiment_start)
						{
							daq_xyz_out[3] = 5.0;  // Start trigger
							experiment_start = false;
							exp_t_last_stim = std::chrono::steady_clock::now();
							exp_t_start = std::chrono::steady_clock::now();
							printf("Started experiment\n");
						}
						else
						{
							daq_xyz_out[3] = 0.0;
							std::chrono::steady_clock::duration duration = std::chrono::steady_clock::now() - exp_t_last_stim;
							float elapsed = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
							if (elapsed >= exp_wait_sec)
							{
								if (exp_n_stim == 0)
								{
									experiment_running = false;
								}
								else
								{
									printf("%i seconds elapsed. Generating pulse on aux2\n", elapsed);
									daq_xyz_out[4] = 5.0;
									exp_n_stim -= 1;
									exp_t_last_stim = std::chrono::steady_clock::now();
								}
							}
							else
							{
								daq_xyz_out[4] = 0.0;
							}
						}
					}

					MotionVector v = { correlation_out[0],
									   correlation_out[1],
									   correlation_out[2],
									   correlation_out[3],
									   correlation_out[4],
									   correlation_out[5],
									   daq_xyz_out[0],
									   daq_xyz_out[1],
									   daq_xyz_out[2],
									   daq_xyz_out[3],
									   daq_xyz_out[4]
					};
					output_queue->enqueue(v);

					if (!dac_output_enabled)  // Zero the correction output if dac_output_enabled is false
					{
						daq_xyz_out[0] = 0;
						daq_xyz_out[1] = 0;
						daq_xyz_out[2] = 0;
					}
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

					total += 1;
					n_wanted += 1;

				}
				else
				{
					acq_buffer->release();
				}
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

		msg_queue = new MotionQueue(512);
		main_running = ATOMIC_VAR_INIT(true);
		running = ATOMIC_VAR_INIT(false);
		mot_thread = std::thread(&MotionWorker::main, this);

		correlation_out = new double[6];
		daq_xyz_out = new double[5];  // Buffer for samples before they are written
		daq_xyz_scale = new double[3]; // Scale factors for DAC channels
		mot_samps_written = new int32[5];  // TODO multi-channel

		memset(daq_xyz_out, 0, 5 * sizeof(double));
		memset(daq_xyz_scale, 1 / 4, 3 * sizeof(double));
		memset(mot_samps_written, 0, 5 * sizeof(int32));

		filters_xyz = new SimpleKalmanFilter[3];
		filter_input_xyz = new Eigen::VectorXd[3];

		output_queue = new MotionResultsQueue(32);
		printf("Motion vector output queue created at %p\n", output_queue);
	}

	// DO NOT access anything thread unsafe from outside main

	bool is_running()
	{
		return running.load();
	}

	void start(int spatial_aline_size, CircAcqBuffer<fftwf_complex>* buffer, int upsample_factor, int* input_dims, int centroid_n_peak,
			   float* spectral_filter_in, float* spatial_filter_in, bool bidirectional, double* filter_e, double* filter_f, double* filter_g, double* filter_q, double* filter_r1, double* filter_r2, double filter_dt, int n_lag)
	{
		MotionMessage msg;
		msg.flag = Start;
		msg.spatial_aline_size = spatial_aline_size;
		msg.circacqbuffer = buffer;
		msg.upsample_factor = upsample_factor;
		msg.input_dims = input_dims;
		msg.centroid_n_peak = centroid_n_peak;
		msg.spatial_filter = spatial_filter_in;
		msg.spectral_filter = spectral_filter_in;
		msg.bidirectional = bidirectional;
		msg.filter_e = filter_e;
		msg.filter_f = filter_f;
		msg.filter_g = filter_g;
		msg.filter_r1 = filter_r1;
		msg.filter_r2 = filter_r2;
		msg.filter_q = filter_q;
		msg.filter_dt = filter_dt;
		msg.n_lag = n_lag;
		msg_queue->enqueue(msg);
	}

	void updateParameters(int centroid_n_peak, float* spectral_filter_in, float* spatial_filter_in, bool bidirectional,
		double* filter_e, double* filter_f, double* filter_g, double* filter_q, double* filter_r1, double* filter_r2, double filter_dt)
	{
		MotionMessage msg;
		msg.flag = UpdateParameters;
		msg.centroid_n_peak = centroid_n_peak;
		msg.spatial_filter = spatial_filter_in;
		msg.spectral_filter = spectral_filter_in;
		msg.bidirectional = bidirectional;
		msg.filter_e = filter_e;
		msg.filter_f = filter_f;
		msg.filter_g = filter_g;
		msg.filter_r1 = filter_r1;
		msg.filter_r2 = filter_r2;
		msg.filter_q = filter_q;
		msg.filter_dt = filter_dt;
		msg_queue->enqueue(msg);
	}

	void configureOutput(const char* ao_dx_ch, const char* ao_dy_ch, const char* ao_dz_ch, double* scale_xyz, bool enabled)
	{
		MotionMessage msg;
		msg.flag = ConfigOutput;
		msg.ao_dx_ch = ao_dx_ch;
		msg.ao_dy_ch = ao_dy_ch;
		msg.ao_dz_ch = ao_dz_ch;
		msg.scale_xyz = scale_xyz;
		msg.output_enabled = enabled;
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

	// An experiment consists of: [start trigger] followed by [wait] [stim trigger] x nStim ... [wait]
	void runExperiment(int nStim, int waitSeconds)
	{
		MotionMessage msg;
		msg.flag = RunExperiment;
		msg.exp_n_stim = nStim;
		msg.exp_wait_seconds = waitSeconds;
		msg_queue->enqueue(msg);
	}

	int grabMotionVector(double* out)
	{
		MotionVector v;
		if (output_queue->dequeue(v))
		{
			out[0] = v.x;
			out[1] = v.y;
			out[2] = v.z;
			out[3] = v.dx;
			out[4] = v.dy;
			out[5] = v.dz;
			out[6] = v.filtered_x;
			out[7] = v.filtered_y;
			out[8] = v.filtered_z;
			out[9] = v.aux1;
			out[10] = v.aux2;
			return 0;
		}
		else
		{
			return -1;
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

	~MotionWorker()
	{
		delete[] correlation_out;
		delete[] daq_xyz_out;
		delete[] daq_xyz_scale;
		delete[] mot_samps_written;
		delete[] filters_xyz;
		delete[] filter_input_xyz;

		delete output_queue;
	}

};
