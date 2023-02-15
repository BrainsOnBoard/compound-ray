Windows Install Notes
=====================

Note that under Windows only the CompoundRay library is currently built.

Please refer and familiarise yourself with the [Linux install guide](https://github.com/BrainsOnBoard/compound-ray/blob/master/docs/indepth-install-notes.md) first.
While it is not appropriate for the operating system you are operating under, it provides useful context.

This file gives you the cliff-notes on installing CompoundRay under Windows using CMake and Visual Studio.

CMake can be installed for Windows [here](https://cmake.org/install/). NVidia CUDA and OptiX can be installed [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [here](https://developer.nvidia.com/designworks/optix/download)

Once OptiX has been installed, follow the build and compilation instructions found in `optix/SDK/INSTALL-WIN.txt`. The same process will be used when building and installing CompoundRay. Visual Studio Community Edition can be found [here](https://visualstudio.microsoft.com/vs/community/). It is a free and accessible version of Visual Studio. Ensure that when installing you include the C/C++ development environment.

With OptiX build correctly, you can now clone the Windows branch of CompoundRay (merge into main pending), named ["windows-fixes"](https://github.com/BrainsOnBoard/compound-ray/tree/windows-fixes).
Once that has been cloned, follow the same process you followed for building the OptiX samples, but instead of running from the "SDK" folder, build from the "compound-ray" folder (containing "libEyeRenderer3" and "sutil").

Ensure that you read any CMake warnings and errors that might arise when generating the project - in particular, you will likely have to specify the `ARCH_INT` variable to your GPU [architecture](https://github.com/BrainsOnBoard/compound-ray/blob/windows-fixes/docs/indepth-install-notes.md#compiling-compoundray) and the `OPTIX_INSTALL_DIR` to point to the [root directory (the one including "SDK", "include", and "doc")](https://github.com/BrainsOnBoard/compound-ray/blob/windows-fixes/docs/indepth-install-notes.md#troubleshooting---building) of the NVidia OptiX SDK.

Once you've configured the project, you will be able to open it using Visual Studio. From there you should be able to compile it using the build button (green arrow) at the top of the UI.
The build will not launch anything, as it is building the CompoundRay library rather than any particular tool.

To generate the actual CompoundRay direct link library (EyeRenderer3.dll), you must build a release version of the project - you can do this by changing "Debug" to "Release" on the build options to the left of the build button.
Once built, the library can be found within the `bin/Release` folder within the project folder (containing "Eye-Renderer-Three.sln") that CMake generated.
