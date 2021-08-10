#include "RealtimeOCTController.h"
#include <Windows.h>
#include <string>
#include <thread>
#include <chrono>
#include "fftw3.h"

extern "C"
{

	__declspec(dllexport) RealtimeOCTController* RTOCT_open(const char* cam_name, const char* ao_x_ch, const char* ao_y_ch, const char* ao_lt_ch, const char* ao_ft_ch)
	{
		RealtimeOCTController* controller = new RealtimeOCTController();
		controller->open(cam_name, ao_x_ch, ao_y_ch, ao_lt_ch, ao_ft_ch);
		return controller;
	}

	__declspec(dllexport) void RTOCT_configure(RealtimeOCTController* controller, int dac_rate, int aline_size, int roi_offset, int roi_size, int number_of_alines, int number_of_buffers)
	{
		controller->configure(dac_rate, aline_size, roi_offset, roi_size, number_of_alines, number_of_buffers);
	}

	__declspec(dllexport) void RTOCT_setProcessing(RealtimeOCTController* controller, double intpdk, float* apodization_window)
	{
		controller->set_oct_processing(intpdk, apodization_window);
	}

	__declspec(dllexport) void RTOCT_setScan(RealtimeOCTController* controller, double* x, double* y, double* linetrigger, double* frametrigger, int n)
	{
		controller->set_scan_signals(x, y, linetrigger, frametrigger, n);
	}

	__declspec(dllexport) void RTOCT_startScan(RealtimeOCTController* controller)
	{
		controller->start_scan();
	}

	__declspec(dllexport) void RTOCT_stopScan(RealtimeOCTController* controller)
	{
		controller->stop_scan();
	}

	__declspec(dllexport) void RTOCT_startSave(RealtimeOCTController* controller, const char* fname, int max_bytes)
	{
		controller->start_save(fname, max_bytes);
	}

	_declspec(dllexport) void RTOCT_saveN(RealtimeOCTController* controller, const char* fname, int max_bytes, int n_to_save)
	{
		controller->save_n(fname, max_bytes, n_to_save);
	}

	_declspec(dllexport) int RTOCT_grabFrame(RealtimeOCTController* controller, fftwf_complex* out)
	{
		return controller->grab_frame(out);
	}

	_declspec(dllexport) void RTOCT_grabSpectrum(RealtimeOCTController* controller, float* out)
	{
		controller->grab_spectrum(out);
	}

	__declspec(dllexport) void RTOCT_stopSave(RealtimeOCTController* controller)
	{
		controller->stop_save();
	}

	__declspec(dllexport) void RTOCT_close(RealtimeOCTController* controller)
	{
		controller->close();
		delete controller;
	}

	// -- MOTION QUANTIFICATION -------------------------------

	__declspec(dllexport) void RTOCT_start_motion_output(RealtimeOCTController* controller, int* input_dims, int upsample_factor, int centroid_n_peak, float* window)
	{
		controller->start_motion_output(input_dims, upsample_factor, centroid_n_peak, window);
	}

	__declspec(dllexport) void RTOCT_stop_motion_output(RealtimeOCTController* controller)
	{
		controller->stop_motion_output();
	}

	__declspec(dllexport) void RTOCT_update_motion_reference(RealtimeOCTController* controller)
	{
		controller->update_motion_reference();
	}

	__declspec(dllexport) void RTOCT_grab_motion_correlogram(RealtimeOCTController* controller, fftwf_complex* out)
	{
		controller->grab_motion_correlogram(out);
	}

	__declspec(dllexport) void RTOCT_grab_motion_frame(RealtimeOCTController* controller, fftwf_complex* out)
	{
		controller->grab_motion_frame(out);
	}

	__declspec(dllexport) bool RTOCT_grab_motion_vector(RealtimeOCTController* controller, double* out)
	{
		return controller->grab_motion_vector(out);
	}

}