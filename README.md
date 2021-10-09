# CompoundRay: A hardware-accellerated compound eye perspective renderer

[CompoundRay](https://www.biorxiv.org/content/10.1101/2021.09.20.461066v1) is a hardware-accellerated compound eye perspective rendering system and API built on top of the NVidia OptiX raytracing engine.


## Building
To build the software you must first install [NVidia CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) and
the [Nvidia OptiX framework](https://developer.nvidia.com/designworks/optix/download). Once these have been installed, follow the
build instructions in eye-renderer/build/&lt;make or cmake&gt;/readme.txt

This software was built and tested first on Manjaro Linux (so the build rules should still work under this), and then on Ubuntu 20.04.2 LTS.
It uses OptiX SDK 7.2.0 or higher, and has been tested on 7.2.0. It requires Cuda version 5.0 or higher.

In order for the OptiX SDK to work, you will need a new version of the Nvidia graphics drivers (460 or higher is recommended).

## Creating Environments
The eye renderer ingests [glTF](https://github.com/KhronosGroup/glTF) format files with extra tags appended to the "extras" tag
within any given camera's definition. An example of this can be seen in [eye-renderer/data/test-scene.gltf](https://github.com/ManganLab/eye-renderer/blob/master/data/test-scene/test-scene.gltf#L229)
on line 229, in which a camera is given the additional properties of being compound with the `"compound-eye" : "true"` property,
the projection schema selected with the `"compound-projection" : "spherical_orientationwise"` property, and a link to the relative 
.eye eye descriptor file ([test.eye](https://github.com/ManganLab/eye-renderer/blob/master/data/test-scene/test.eye)) added with the
`"compound-structure" : "test.eye"` property. These properties can be set manually by editing the .glTF file format (which is in 
human-readable [json](https://docs.fileformat.com/web/json/), or edited directly on each object (and camera) using the properties
panel in [Blender3D](https://www.blender.org/).

Eye structure files can be made as CSVs with the .eye file extension, contents defined [here](https://github.com/ManganLab/eye-renderer/blob/master/data/eyes/eye-specification.txt).

## Using the Renderer
A quick way to validate your model is to load it using the stand-alone rendering tool, which will be placed into the eye-renderer/build/&lt;ninja or make&gt;/bin folder.
Running this tool with the `-h` switch will give instructions to usage.

However, the more efficient way of using the tool is by using the underlying API directly. This is best achieved through use of 
Python and the [ctypes binding library](https://docs.python.org/3/library/ctypes.html) - the primary example of using this (along
with the bundled python [eyeRendererHelperFunctions](https://github.com/ManganLab/eye-renderer/blob/master/python-examples/eyeRendererHelperFunctions.py))
can be found [here](https://github.com/ManganLab/eye-renderer/blob/master/python-examples/primary-example.py), although other useful
examples (including those from the [CompoundRay paper](https://www.biorxiv.org/content/10.1101/2021.09.20.461066v1) can be found in the [eye-renderer/python-examples](https://github.com/ManganLab/eye-renderer/tree/master/python-examples)
folder, or through the [minimumSampleRateFinder tool](https://github.com/ManganLab/eye-renderer/blob/master/data/tools/minimumSampleRateFinder.py)
script.
