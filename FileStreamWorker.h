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
#include <fstream>

enum FileStreamMessageFlag
{
	StartStream = 1 << 1,
	StopStream = 1 << 2,
	StreamN = 1 << 3,
};

DEFINE_ENUM_FLAG_OPERATORS(FileStreamMessageFlag);

enum FileStreamType
{
	FSTREAM_TYPE_TIF = 1 << 1,
	FSTREAM_TYPE_NPY = 1 << 2,
	FSTREAM_TYPE_MAT = 1 << 3,
};

DEFINE_ENUM_FLAG_OPERATORS(FileStreamType);

struct FileStreamMessage
{
	FileStreamMessageFlag flag;
	const char* fname;
	int fsize;
	FileStreamType ftype;
	CircAcqBuffer<fftwf_complex>* circacqbuffer;
	int aline_size;
	int number_of_alines;
	int roi_size;  // Number of voxels in each A-line
	int n_to_stream;  // Number of frames to save for numbered stream
};

typedef spsc_bounded_queue_t<FileStreamMessage> FstreamQueue;

class FileStreamWorker final
{

protected:

	int id;

	std::thread fstream_thread;

	FstreamQueue* msg_queue;

	std::atomic_bool main_running;
	std::atomic_bool streaming;

	int frames_in_current_file;

	const char* file_name;
	int file_name_inc;
	int file_max_bytes;
	FileStreamType file_type;
	CircAcqBuffer<fftwf_complex>* acq_buffer;

	int aline_size;
	int number_of_alines;
	int roi_size;
	int n_to_stream;

	inline void recv_msg()
	{
		FileStreamMessage msg;
		if (msg_queue->dequeue(msg))
		{
			if (msg.flag & StartStream)
			{
				if (!streaming.load())  // Ignore if already streaming
				{
					streaming.store(true);
	
					acq_buffer = msg.circacqbuffer;
					
					n_to_stream = -1;

					file_name = msg.fname;
					file_name_inc = 0;
					file_type = msg.ftype;

					aline_size = msg.aline_size;
					number_of_alines = msg.number_of_alines;
					roi_size = msg.roi_size;

					printf("Continuous save to file(s) %s...\n", file_name);

				}
			}
			if (msg.flag & StreamN)
			{
				printf("Numbered save: streaming %i frames to file %s.\n", msg.n_to_stream, file_name);
				n_to_stream = msg.n_to_stream;
			}
			if (msg.flag & StopStream)
			{
				if (streaming.load())
				{
					streaming.store(false);
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
		printf("FileStreamWorker main() running...\n");

		bool fopen = false;
		int n_wanted = 0;
		int n_got;
		fftwf_complex* f;

		std::ofstream fout;

		while (main_running.load() == true)
		{
			this->recv_msg();
			if (streaming.load())
			{
				n_got = acq_buffer->lock_out_wait(n_wanted, &f);

				if (n_wanted == n_got)
				{
					n_wanted += 1;

					if (!fopen)
					{
						// Open file
						char fname[512];
						sprintf(fname, "%s", file_name);  // TODO increment for large files
						printf("Opened file %s\n", fname);
						fout = std::ofstream(fname, std::ios::out | std::ios::binary);
						frames_in_current_file = 0;
						fopen = true;
					}
					else
					{
						if (n_to_stream == -1)  // Indefinite stream
						{
							// Append to file
							fout.write((char*)f, number_of_alines * roi_size * sizeof(fftwf_complex));
							frames_in_current_file += 1;
						}
						else if (n_to_stream - frames_in_current_file > 0)  // Streaming n
						{
							fout.write((char*)f, number_of_alines * roi_size * sizeof(fftwf_complex));
							frames_in_current_file += 1;
						}
						else
						{
							// Close file, stop streaming
							printf("Closing file %s\n", file_name);
							fout.close();
							fopen = false;
						}
					}

				}
				else  // Dropped frame, since we have fallen behind, get the latest next time
				{
					printf("Dropped frame!\n");
					n_wanted = acq_buffer->get_count();
				}
				acq_buffer->release();
			}
			else
			{
				if (fopen)
				{
					// Close file
					printf("Closing file %s\n", file_name);
					fout.close();
					fopen = false;
				}
			}
		}
	}

public:

	FileStreamWorker()
	{
		msg_queue = new FstreamQueue(65536);
		main_running = ATOMIC_VAR_INIT(false);
		streaming = ATOMIC_VAR_INIT(false);
	}

	FileStreamWorker(int thread_id)
	{

		id = thread_id;

		msg_queue = new FstreamQueue(65536);
		main_running = ATOMIC_VAR_INIT(true);
		streaming = ATOMIC_VAR_INIT(false);
		fstream_thread = std::thread(&FileStreamWorker::main, this);

	}

	// DO NOT access non-atomics from outside main()

	bool is_streaming()
	{
		return streaming.load();
	}

	void start_streaming(const char* fname, int max_bytes, FileStreamType ftype, CircAcqBuffer<fftwf_complex>* buffer, int aline_size, int number_of_alines, int roi_size)
	{
		FileStreamMessage msg;
		msg.flag = StartStream;
		msg.fname = fname;
		msg.fsize = max_bytes;
		msg.circacqbuffer = buffer;
		msg.aline_size = aline_size;
		msg.number_of_alines = number_of_alines;
		msg.roi_size = roi_size;
		msg_queue->enqueue(msg);
	}

	void start_streaming(const char* fname, int max_bytes, FileStreamType ftype, CircAcqBuffer<fftwf_complex>* buffer, int aline_size, int number_of_alines, int roi_size, int n_to_stream)
	{
		FileStreamMessage msg;
		msg.flag = StartStream | StreamN;
		msg.fname = fname;
		msg.fsize = max_bytes;
		msg.circacqbuffer = buffer;
		msg.aline_size = aline_size;
		msg.number_of_alines = number_of_alines;
		msg.roi_size = roi_size;
		msg.n_to_stream = n_to_stream;
		msg_queue->enqueue(msg);
	}

	void stop_streaming()
	{
		FileStreamMessage msg;
		msg.flag = StopStream;
		msg_queue->enqueue(msg);
	}

	void terminate()
	{
		main_running.store(false);
		fstream_thread.join();
	}

};
