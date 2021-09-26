#include "RealtimeOCTController.h"
#include "PhaseCorrelationPlan3D.h"

#define WINVER 0x0502
#define _WIN32_WINNT 0x0502

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

	__declspec(dllexport) bool RTOCT_is_scanning(RealtimeOCTController* controller)
	{
		return controller->is_scanning();
	}

	__declspec(dllexport) bool RTOCT_is_ready_to_scan(RealtimeOCTController* controller)
	{
		return controller->is_ready_to_scan();
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

	__declspec(dllexport) void RTOCT_start_motion_output(RealtimeOCTController* controller, int* input_dims, double* scale_xyz, int upsample_factor, int centroid_n_peak,
														 float* spectral_filter, float* spatial_filter, bool bidirectional, double* filter_d, double* filter_g, double* filter_q, double* filter_r)
	{
		controller->start_motion_output(input_dims, scale_xyz, upsample_factor, centroid_n_peak, spectral_filter, spatial_filter, bidirectional, filter_d, filter_g, filter_q, filter_r);
	}

	__declspec(dllexport) void RTOCT_stop_motion_output(RealtimeOCTController* controller)
	{
		controller->stop_motion_output();
	}

	__declspec(dllexport) void RTOCT_update_motion_parameters(RealtimeOCTController* controller, double* scale_xyz, int centroid_n_peak, float* spectral_filter, float* spatial_filter, bool bidirectional,
															  double* filter_d, double* filter_g, double* filter_q, double* filter_r)
	{
		controller->update_motion_parameters(scale_xyz, centroid_n_peak, spectral_filter, spatial_filter, bidirectional, filter_d, filter_g, filter_q, filter_r);
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

	__declspec(dllexport) int RTOCT_grab_motion_vector(RealtimeOCTController* controller, double* out)
	{
		return controller->grab_motion_vector(out);
	}

	// -- PHASE CORR PLAN -------------------------------------

	__declspec(dllexport) PhaseCorrelationPlan3D* PCPLAN3D_create(int* input_dims, int upsample, int npeak_centroid, float* spectral_filter_3d, float* spatial_filter_3d, bool bidirectional)
	{
		PhaseCorrelationPlan3D* plan = new PhaseCorrelationPlan3D(input_dims, upsample, npeak_centroid, spectral_filter_3d, spatial_filter_3d, bidirectional);
		return plan;
	}

	__declspec(dllexport) void PCPLAN3D_close(PhaseCorrelationPlan3D* plan)
	{
		delete plan;
	}

	__declspec(dllexport) void PCPLAN3D_set_reference(PhaseCorrelationPlan3D* plan, fftwf_complex* t0)
	{
		plan->setReference(t0);
	}

	__declspec(dllexport) void PCPLAN3D_get_displacement(PhaseCorrelationPlan3D* plan, fftwf_complex* tn, double* out)
	{
		plan->getDisplacement(tn, out);
	}

	__declspec(dllexport) void PCPLAN3D_get_r(PhaseCorrelationPlan3D* plan, fftwf_complex* out)
	{
		fftwf_complex* f = plan->get_r();
		memcpy(out, f, plan->get_frame_size() * sizeof(fftwf_complex));
	}

	__declspec(dllexport) void PCPLAN3D_get_R(PhaseCorrelationPlan3D* plan, fftwf_complex* out)
	{
		fftwf_complex* f = plan->get_R();
		memcpy(out, f, plan->get_frame_size() * sizeof(fftwf_complex));
	}

	__declspec(dllexport) void PCPLAN3D_get_tn(PhaseCorrelationPlan3D* plan, fftwf_complex* out)
	{
		fftwf_complex* f = plan->get_tn();
		memcpy(out, f, plan->get_frame_size() * sizeof(fftwf_complex));
	}

	__declspec(dllexport) void PCPLAN3D_get_t0(PhaseCorrelationPlan3D* plan, fftwf_complex* out)
	{
		fftwf_complex* f = plan->get_t0();
		memcpy(out, f, plan->get_frame_size() * sizeof(fftwf_complex));
	}

}