# RT-OCT-Mot-Corr

Requirements:
- [Qt 5.9.2](https://download.qt.io/archive/qt/5.9/5.9.2/)
- [Visual Studio 2017 Community Edition](https://visualstudio.microsoft.com/vs/older-downloads/), C++ Build Tools
   - The following libraries must be available to build the fastnisdoct library:
      - National Instruments [IMAQ](https://www.ni.com/en-us/support/downloads/drivers/download.vision-acquisition-software.html#409847) and [NI-DAQmx](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#445931)
      - [FFTW](http://www.fftw.org/install/windows.html)
- [Python 3.6.8](https://www.python.org/downloads/release/python-368/)

How to run this project:
- Compile the Visual Studio project (VS2019 recommended). FFTW is required.
- Install required Python packages, including PyQt5==5.15.1 and pyqtgraph==0.11.1.
- Modify NI channel IDs
- Run Realtime3DGUI.py
