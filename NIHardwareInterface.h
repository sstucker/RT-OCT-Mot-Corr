#pragma once

#include <Windows.h>
#include <string>
#include "niimaq.h"
#include "NIDAQmx.h"

class NIHardwareInterface
{

private:

    int err;  // Error state of hardware interface

    bool opened;  // If configured and connected to hardware
    bool scanning;  // If actively scanning and grabbing frames

    // IMAQ

    SESSION_ID session_id;  // IMAQ session ID
    BUFLIST_ID buflist_id;  // IMAQ buflist ID
    INTERFACE_ID interface_id;  // IMAQ interface ID
    TaskHandle scan_task;  // NI-DAQ task handle

    Int32 halfwidth; // A-line size / 2 + 1
    Int32 acqWinWidth;  // A-line size
    Int32 acqWinHeight;  // Number of A-lines per buffer
    Int32 bytesPerPixel;
    int bufferSize;  // Equivalent to acqWinWidth * acqWinHeight * bytesPerPixel
    int numberOfBuffers;  // Number of IMAQ ring buffer elements
    int bufferNumber;  // Cumulative acquired IMAQ ring buffer element

    uInt32 examined_number;  // Cumulative number of the currently examined IMAQ ring buffer element

    Int8** imaq_buffers;  // uint16 ring buffer elements managed by IMAQ

    // NI-DAQ

    int dacRate;  // The output rate of the DAC used to drive the scan pattern
    double* concatenated_scansig;  // Pointer to buffer of scansignals appended end to end
    int scansig_n; // Number of samples in each of the 4 scan signals
    int32* samples_written;  // Returned by NI DAQ after samples are written


    int setup_buffers(UINT32 aline_size, UINT32 number_of_alines, UINT32 number_of_buffers)
    {
        err = imgSetAttribute2(session_id, IMG_ATTR_ACQWINDOW_TOP, 0);
        err = imgSetAttribute2(session_id, IMG_ATTR_ACQWINDOW_LEFT, 0);
        err = imgSetAttribute2(session_id, IMG_ATTR_ACQWINDOW_HEIGHT, number_of_alines);
        err = imgSetAttribute2(session_id, IMG_ATTR_ACQWINDOW_WIDTH, aline_size);
        if (err != 0)
        {
            return err;
        }
        // Confirm the change by getting the attributes
        err = imgGetAttribute(session_id, IMG_ATTR_ROI_WIDTH, &acqWinWidth);
        err = imgGetAttribute(session_id, IMG_ATTR_ROI_HEIGHT, &acqWinHeight);
        err = imgGetAttribute(session_id, IMG_ATTR_BYTESPERPIXEL, &bytesPerPixel);
        if (err != 0)
        {
            return err;
        }

        bufferSize = acqWinWidth * acqWinHeight * bytesPerPixel;
        halfwidth = acqWinWidth / 2 + 1;

        numberOfBuffers = number_of_buffers;

        err = imgCreateBufList(number_of_buffers, &buflist_id);
        imaq_buffers = new Int8 * [number_of_buffers];

        int bufCmd;
        for (int i = 0; i < numberOfBuffers; i++)
        {
            err = imgCreateBuffer(session_id, FALSE, bufferSize, (void**)&imaq_buffers[i]);
            if (err != 0)
            {
                return err;
            }
            err = imgSetBufferElement2(buflist_id, i, IMG_BUFF_ADDRESS, imaq_buffers[i]);
            if (err != 0)
            {
                return err;
            }
            err = imgSetBufferElement2(buflist_id, i, IMG_BUFF_SIZE, bufferSize);
            if (err != 0)
            {
                return err;
            }
            bufCmd = (i == (number_of_buffers - 1)) ? IMG_CMD_LOOP : IMG_CMD_NEXT;
            if (err != 0)
            {
                return err;
            }
            err = imgSetBufferElement2(buflist_id, i, IMG_BUFF_COMMAND, bufCmd);
            if (err != 0)
            {
                return err;
            }
        }
        err = imgMemLock(buflist_id);
        if (err != 0)
        {
            return err;
        }
        err = imgSessionConfigure(session_id, buflist_id);
        if (err != 0)
        {
            return err;
        }
        return err;
    }


    int open_imaq_interface(const char* cameraName, int aline_size, int number_of_alines, int number_of_buffers)
    {
        err = imgInterfaceOpen(cameraName, &interface_id);
        if (err != 0)
        {
            return err;
        }
        printf("IMAQ session opened with camera %s\n", cameraName);
        err = imgSessionOpen(interface_id, &session_id);
        // Configure the frame acquisition to be triggered by the TTL1 line
        err = imgSetAttribute2(session_id, IMG_ATTR_EXT_TRIG_LINE_FILTER, true);
        // Frame trigger TTL1
        err = imgSessionTriggerConfigure2(session_id, IMG_SIGNAL_EXTERNAL, IMG_EXT_TRIG1, IMG_TRIG_POLAR_ACTIVEH, 1000, IMG_TRIG_ACTION_BUFFER);
        // Frame trigger output TTL2
        err = imgSessionTriggerDrive2(session_id, IMG_SIGNAL_EXTERNAL, IMG_EXT_TRIG2, IMG_TRIG_POLAR_ACTIVEH, IMG_TRIG_DRIVE_FRAME_START);

        err = setup_buffers(aline_size, number_of_alines, number_of_buffers);

        return err;
    }


    int close_imaq_interface()
    {
        err = imgMemUnlock(buflist_id);
        for (int i = 0; i < numberOfBuffers; i++)
        {
            if (imaq_buffers[i] != NULL)
            {
                err = imgDisposeBuffer(imaq_buffers[i]);
            }
        }
        err = imgDisposeBufList(buflist_id, FALSE);
        err = imgClose(session_id, TRUE);
        err = imgClose(interface_id, TRUE);

        return err;
    }


    int open_nidaq_interface(const char* aoScanX, const char* aoScanY, const char* aoLineTrigger, const char* aoFrameTrigger, int dac_rate)
    {
        err = DAQmxCreateTask("scan", &scan_task);
        err = DAQmxCreateAOVoltageChan(scan_task, aoScanX, "", -10, 10, DAQmx_Val_Volts, NULL);
        err = DAQmxCreateAOVoltageChan(scan_task, aoScanY, "", -10, 10, DAQmx_Val_Volts, NULL);
        err = DAQmxCreateAOVoltageChan(scan_task, aoLineTrigger, "", -10, 10, DAQmx_Val_Volts, NULL);
        err = DAQmxCreateAOVoltageChan(scan_task, aoFrameTrigger, "", -10, 10, DAQmx_Val_Volts, NULL);

        if (err != 0)
        {
            printf("DAQmx failed to create task with channels %s, %s, %s, %s:\n", aoScanX, aoScanY, aoLineTrigger, aoFrameTrigger);
            char* buf = new char[512];
            DAQmxGetErrorString(err, buf, 512);
            printf(buf);
            printf("\n");
            delete[] buf;
            return err;
        }
        else
        {
            err = DAQmxSetWriteRegenMode(scan_task, DAQmx_Val_AllowRegen);
            err = DAQmxSetSampTimingType(scan_task, DAQmx_Val_SampleClock);
            err = DAQmxCfgSampClkTiming(scan_task, NULL, dac_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, NULL);
            if (err != 0)
            {
                printf("DAQmx failed to program the task:\n");
                char* buf = new char[512];
                DAQmxGetErrorString(err, buf, 512);
                printf(buf);
                printf("\n");
                delete[] buf;
                return err;
            }
            dacRate = dac_rate;
        }
        return err;
    }


    void close_nidaq_interface()
    {
        DAQmxClearTask(scan_task);
        delete[] concatenated_scansig;
        delete[] samples_written;
    }

    inline void set_scan_signals(double* x, double* y, double* linetrigger, double* frametrigger, int n)
    {
        // Assign buffers for scan pattern
        delete[] samples_written;
        delete[] concatenated_scansig;
        samples_written = new int32[4];
        concatenated_scansig = new double[4 * n];
        memcpy(concatenated_scansig + 0, x, sizeof(double) * n);
        memcpy(concatenated_scansig + n, y, sizeof(double) * n);
        memcpy(concatenated_scansig + 2 * n, linetrigger, sizeof(double) * n);
        memcpy(concatenated_scansig + 3 * n, frametrigger, sizeof(double) * n);

        if (n != this->scansig_n)  // If buffer size needs to change
        {
            if (scanning)  // If task is running, need to stop, change buffer size, and restart it
            {
                err = DAQmxStopTask(scan_task);
                err = DAQmxCfgOutputBuffer(scan_task, n);
                err = DAQmxWriteAnalogF64(scan_task, n, false, 1000, DAQmx_Val_GroupByChannel, concatenated_scansig, samples_written, NULL);
                err = DAQmxStartTask(scan_task);

            }
            else  // If task isn't running, just buffer new samples without starting
            {
                err = DAQmxCfgOutputBuffer(scan_task, n);
                err = DAQmxWriteAnalogF64(scan_task, n, false, 1000, DAQmx_Val_GroupByChannel, concatenated_scansig, samples_written, NULL);
            }
        }
        else
        {
            err = DAQmxWriteAnalogF64(scan_task, n, false, 1000, DAQmx_Val_GroupByChannel, concatenated_scansig, samples_written, NULL);
        }

        if (err != 0)
        {
            printf("DAQmx failed to set scan signals:\n");
            char* buf = new char[512];
            DAQmxGetErrorString(err, buf, 512);
            printf(buf);
            printf("\n");
            delete[] buf;
        }

        scansig_n = n;  // Set property to new n

    }


public:

	NIHardwareInterface()
	{
        err = 0;
        opened = 0;
        scanning = 0;
        session_id = NULL;
        buflist_id = NULL;
        interface_id = NULL;
        scan_task = NULL;
        halfwidth = 0;
        acqWinWidth = 0;
        acqWinHeight = 0;
        bytesPerPixel = 0;
        bufferSize = 0;
        numberOfBuffers = 0;
        bufferNumber = 0;
        examined_number = -1;
        imaq_buffers = NULL;  // uint16 ring buffer elements managed by IMAQ
        dacRate = 0;
        concatenated_scansig = NULL;
        scansig_n = 0;
        samples_written = 0;
	}

    int open(const char* camera_name, const char* x_ch_name, const char* y_ch_name, const char* line_trig_ch_name, const char* frame_trig_ch_name, int dac_rate, int aline_size, int number_of_alines, int number_of_buffers)
    {
        // printf("NIHardwareInterface opening with ch %s %s %s %s %s\n", camera_name, x_ch_name, y_ch_name, line_trig_ch_name, frame_trig_ch_name);
        if (!opened)
        {
            err = open_imaq_interface(camera_name, aline_size, number_of_alines, number_of_buffers);
            if (err != 0)
            {
                char* buf = new char[512];
                imgShowError(err, buf);
                printf(buf);
                printf("\n");
                delete[] buf;
                return err;
            }
            open_nidaq_interface(x_ch_name, y_ch_name, line_trig_ch_name, frame_trig_ch_name, dac_rate);
            if (err == 0)
            {
                opened = true;
                return 0;
            }
            else
            {
                char* buf = new char[512];
                DAQmxGetErrorString(err, buf, 512);
                printf(buf);
                printf("\n");
                delete[] buf;
                return err;
            }
        }
        else
        {
            printf("Cannot open: hardware interface already opened.\n");
            return -1;
        }
    }

    int close()
    {
        if (opened)
        {
            close_nidaq_interface();
            close_imaq_interface();
            opened = false;
            return 0;
        }
        else
        {
            printf("Cannot close: hardware interface not opened.\n");
            return -1;
        }
    }

    int start_scan()
    {
        if (!scanning && opened)
        {
            err = DAQmxStartTask(scan_task);
            if (err == 0)
            {
                err = imgSessionStartAcquisition(session_id);
                if (err == 0)
                {
                    scanning = true;
                    return 0;
                }
                else
                {
                    printf("Failed to start scan. IMAQ error:\n");
                    char* buf = new char[512];
                    imgShowError(err, buf);
                    printf(buf);
                    printf("\n");
                    delete[] buf;
                    return err;
                }
            }
            else
            {
                printf("Failed to start scan. DAQmx error:\n");
                char* buf = new char[512];
                DAQmxGetErrorString(err, buf, 512);
                printf(buf);
                printf("\n");
                delete[] buf;
                return err;
            }
        }
        else
        {
            printf("Cannot start scan!\n");
            return -1;
        }

    }

    int examine_imaq_buffer(uint16_t** raw_frame_addr, int frame_index)
    {
        err = imgSessionExamineBuffer2(session_id, frame_index, &examined_number, (void**)raw_frame_addr);
        if (err == 0)
        {
            return examined_number;
        }
        else if (err == IMG_ERR_TIMO || err == IMG_ERR_TIMEOUT)  // Don't print error if it is timeout
        {
            printf("IMAQ examine buffer timed out trying to get %i\n", frame_index);
            return -1;
        }
        else
        {
            char* buf = new char[512];
            imgShowError(err, buf);
            printf(buf);
            printf("\n");
            delete[] buf;
            return -1;
        }
    }

    void release_imaq_buffer()
    {
        imgSessionReleaseBuffer(session_id);
    }

    int stop_scan()
    {
        if (scanning)
        {
            err = imgSessionStopAcquisition(session_id);
            err = DAQmxStopTask(scan_task);
            if (err == 0)
            {
                scanning = false;
                return 0;
            }
            else
            {
                return err;
            }
        }
        else
        {
            return -1;
        }
    }

    void set_scan_pattern(double* x, double* y, double* linetrigger, double* frametrigger, int scansig_length)
    {
        set_scan_signals(x, y, linetrigger, frametrigger, scansig_length);
    }

    ~NIHardwareInterface()
    {
        // delete[] samples_written;
        // delete[] concatenated_scansig;
    }

};

