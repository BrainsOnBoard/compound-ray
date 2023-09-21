# CompoundRay: A hardware-accellerated compound eye perspective renderer

[CompoundRay](https://elifesciences.org/articles/73893) is a hardware-accellerated compound eye perspective rendering system and API built on top of the NVidia OptiX raytracing engine.


## Building
To build the software you must first install [NVidia CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) and
the [Nvidia OptiX framework](https://developer.nvidia.com/designworks/optix/download). Once these have been installed, follow the
build instructions in compound-ray/build/&lt;make or ninja&gt;/readme.txt

This software was built and tested first on Manjaro Linux (so the build rules should still work under this), and then on Ubuntu 20.04.2 LTS and Ubuntu 20.04.3 LTS.
It uses OptiX SDK 7.2.0 or higher, and has been tested on 7.2.0 and 7.3.0. It requires Cuda version 5.0 or higher and has been tested on version 11.5. The Make build path is preferred over the ninja build path as it has been more thoroughly tested.

In order for the OptiX SDK to work, you will need a new version of the Nvidia graphics drivers (460 or higher is recommended).

An in-depth build guide under Ubuntu 20.04 can be found [here](docs/indepth-install-notes.md).

If you wish to build under Windows, limited support can be found [here](docs/windows-install-notes.md).

## Creating Environments
The eye renderer ingests [glTF](https://github.com/KhronosGroup/glTF) format files with extra tags appended to the "extras" tag
within any given camera's definition. An example of this can be seen in [compound-ray/data/test-scene.gltf](https://github.com/BrainsOnBoard/compound-ray/blob/master/data/test-scene/test-scene.gltf#L229)
on line 229, in which a camera is given the additional properties of being compound with the `"compound-eye" : "true"` property,
the projection schema selected with the `"compound-projection" : "spherical_orientationwise"` property, and a link to the relative 
.eye eye descriptor file ([test.eye](https://github.com/BrainsOnBoard/compound-ray/blob/master/data/test-scene/test.eye)) added with the
`"compound-structure" : "test.eye"` property. These properties can be set manually by editing the .glTF file format (which is in 
human-readable [json](https://docs.fileformat.com/web/json/), or edited directly on each object (and camera) using the properties
panel in [Blender3D](https://www.blender.org/).

Sky shaders (as are all shaders) are pulled in from [shaders.cu](https://github.com/BrainsOnBoard/compound-ray/blob/master/libEyeRenderer3/shaders.cu) in `libEyeRenderer3`. Currently there is one sky shader available, 
`simple_sky`, which can be added as an extra property on the scene itself (see line 19 of the [natural environment stand-in gltf file](https://github.com/BrainsOnBoard/compound-ray/blob/master/data/natural-standin-sky.gltf)).

Eye structure files can be made as CSVs with the .eye file extension, contents defined [here](https://github.com/BrainsOnBoard/compound-ray/blob/master/data/eyes/eye-specification.txt).

## Using the Renderer
A quick way to validate your model is to load it using the stand-alone rendering tool, which will be placed into the compound-ray/build/&lt;ninja or make&gt;/bin folder.
Running this tool with the `-h` switch will give instructions to usage.

However, the more efficient way of using the tool is by using the underlying API directly. This is best achieved through use of 
Python and the [ctypes binding library](https://docs.python.org/3/library/ctypes.html) - the primary example of using this (along
with the bundled python [eyeRendererHelperFunctions](https://github.com/BrainsOnBoard/compound-ray/blob/master/python-examples/eyeRendererHelperFunctions.py))
can be found [here](https://github.com/BrainsOnBoard/compound-ray/blob/master/python-examples/primary-example.py), although other useful
examples (including those from the [CompoundRay paper](https://www.biorxiv.org/content/10.1101/2021.09.20.461066v1) can be found in the [compound-ray/python-examples](https://github.com/BrainsOnBoard/compound-ray/tree/master/python-examples)
folder, or through the [minimumSampleRateFinder tool](https://github.com/BrainsOnBoard/compound-ray/blob/master/data/tools/minimumSampleRateFinder.py)
script.
