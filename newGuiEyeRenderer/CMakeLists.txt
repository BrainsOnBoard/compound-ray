# Set the local resources from this folder (similar to the resources set in all other CMake files)
set(gui_eye_renderer_sources
  gui.cpp
  BasicController.h
  BasicController.cpp
)

# Include the 'add_eye_renderer' macro from the eye renderer and
# then actually add the correct files to the source list so we can use
# eye renderer library commands.
include("../libEyeRenderer3/CMakeIncludes.cmake")
add_eye_renderer("../libEyeRenderer3" gui_eye_renderer_sources)

OPTIX_add_sample_executable( newGuiEyeRenderer target_name
  ${gui_eye_renderer_sources}
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
