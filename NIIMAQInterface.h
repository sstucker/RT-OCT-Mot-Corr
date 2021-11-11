#pragma once

#include <Windows.h>
#include <string>
#include "niimaq.h"

/*
Interface with National Instruments NI-IMAQ library. Configures and carries
out image acquisition. Code is largely based on the ""LLRing" NI-IMAQ Example.
*/
class NIIMAQInterface
{
private:

	int err;

	bool opened;
	bool scanning;

    SESSION_ID session_id;  // IMAQ session ID
    BUFLIST_ID buflist_id;  // IMAQ buflist ID
    INTERFACE_ID interface_id;  // IMAQ interface ID

    Int32 halfwidth; // A-line size / 2 + 1, AKA spatial A-line size
    Int32 acqWinWidth;  // Raw A-line size
    Int32 acqWinHeight;  // Number of A-lines per IMAQ buffer
    Int32 bytesPerPixel;  // Size of each raw image element
    int bufferSize;  // Equivalent to acqWinWidth * acqWinHeight * bytesPerPixel
    int numberOfBuffers;  // Number of IMAQ ring buffer elements
    int bufferNumber;  // Cumulative acquired IMAQ ring buffer element

    uInt32 examined_number;  // Cumulative number of the currently examined IMAQ ring buffer element

    Int8** imaq_buffers;  // uint16 ring buffer elements managed by IMAQ

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
        err = imgGetAttribute(session_id, IMG_ATTR_ROI_WIDTH, &acqWinWidth);
        err = imgGetAttribute(session_id, IMG_ATTR_ROI_HEIGHT, &acqWinHeight);
        err = imgGetAttribute(session_id, IMG_ATTR_BYTESPERPIXEL, &bytesPerPixel);
        if (err != 0)
        {
            return err;
        }
        else if ((acqWinWidth != aline_size) || (acqWinHeight != number_of_alines))
        {
            return -1;
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
            err = imgSetBufferElement2(buflist_id, i, IMG_BUFF_SIZE, bufferSize);
            bufCmd = (i == (number_of_buffers - 1)) ? IMG_CMD_LOOP : IMG_CMD_NEXT;
            err = imgSetBufferElement2(buflist_id, i, IMG_BUFF_COMMAND, bufCmd);
            if (err != 0)
            {
                return err;
            }
        }
        err = imgMemLock(buflist_id);
        err = imgSessionConfigure(session_id, buflist_id);
        if (err != 0)
        {
            return err;
        }
        return err;
    }

public:


    int open(const char* cameraName, int aline_size, int number_of_alines, int number_of_buffers)
    {
        err = imgInterfaceOpen(cameraName, &interface_id);
        if (err != 0)
        {
            return err;
        }
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


    int close()
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

};
