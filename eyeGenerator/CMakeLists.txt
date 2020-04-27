# Set sources using the CMakeIncludes file that all projects use so I only have to change it once
set(local_eye_genereator_sources main.cpp)
include("CMakeIncludes.cmake")
add_eye_generator("." local_eye_genereator_sources)

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( eyeGenerator
  ${local_eye_genereator_sources}
  )
