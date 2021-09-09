#pragma once

#include <atomic>
#include <condition_variable>

class Barrier
{
	protected:
		std::condition_variable _cv;
		std::atomic_int _arrived;
		int _count;

	public:

		Barrier()
		{
			_count = 0;
			_arrived = 0;
		}

		Barrier(int count)
		{
			_count = count;
			_arrived = 0;
		}

		wait_for(int count)

};
