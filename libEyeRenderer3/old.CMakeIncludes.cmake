# This macro allows all the relevant eye renderer
# files to be added to any other project by appending
# them to the given "sources" list.
#
# It also takes a relative path, which for most projects
# residing in their own folder alongside libEyeRenderer3
# will be "../libEyeRenderer3"

macro(add_eye_renderer relative_path sources)
  include_directories(${relative_path})
  set(eye_renderer_sources
      ${relative_path}/libEyeRenderer.h
      ${relative_path}/libEyeRenderer.cpp
      ${relative_path}/GlobalParameters.h
      ${relative_path}/MulticamScene.h
      ${relative_path}/MulticamScene.cpp
      ${relative_path}/cameras/GenericCamera.h
      ${relative_path}/cameras/GenericCamera.cpp
      ${relative_path}/cameras/PerspectiveCamera.h
      ${relative_path}/cameras/PerspectiveCamera.cpp
      ${relative_path}/cameras/PanoramicCamera.h
      ${relative_path}/cameras/PanoramicCamera.cpp
      ${relative_path}/cameras/OrthographicCamera.h
      ${relative_path}/cameras/OrthographicCamera.cpp
      ${relative_path}/cameras/CompoundEye.h
      ${relative_path}/cameras/CompoundEye.cpp
      ${relative_path}/cameras/DataRecordCamera.h
      ${relative_path}/shaders.cu
     )
  list(APPEND ${sources} ${eye_renderer_sources})
endmacro()
