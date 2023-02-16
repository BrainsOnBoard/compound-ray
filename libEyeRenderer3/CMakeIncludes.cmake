# This macro allows all the relevant eye renderer
# files to be added to any other project by appending
# them to the given "sources" list.
#
# It also takes a relative path, which for most projects
# residing in their own folder alongside libEyeRenderer3
# will be "../libEyeRenderer3"

#macro(add_eye_renderer relative_path sources)
#  include_directories(${relative_path})
#  set(eye_renderer_sources
#      ${relative_path}/libEyeRenderer.h
#      ${relative_path}/libEyeRenderer.cpp
#      ${relative_path}/GlobalParameters.h
#      ${relative_path}/MulticamScene.h
#      ${relative_path}/MulticamScene.cpp
#      ${relative_path}/cameras/GenericCamera.h
#      ${relative_path}/cameras/GenericCamera.cpp
#      ${relative_path}/cameras/PerspectiveCamera.h
#      ${relative_path}/cameras/PerspectiveCamera.cpp
#      ${relative_path}/cameras/PanoramicCamera.h
#      ${relative_path}/cameras/PanoramicCamera.cpp
#      ${relative_path}/cameras/OrthographicCamera.h
#      ${relative_path}/cameras/OrthographicCamera.cpp
#      ${relative_path}/cameras/CompoundEye.h
#      ${relative_path}/cameras/CompoundEye.cpp
#      ${relative_path}/cameras/DataRecordCamera.h
#      ${relative_path}/shaders.cu
#     )
#  list(APPEND ${sources} ${eye_renderer_sources})
#endmacro()


####### NEW:

set(sources
    libEyeRenderer.h
    libEyeRenderer.cpp
    GlobalParameters.h
    MulticamScene.h
    MulticamScene.cpp
    cameras/GenericCamera.h
    cameras/GenericCamera.cpp
    cameras/PerspectiveCamera.h
    cameras/PerspectiveCamera.cpp
    cameras/PanoramicCamera.h
    cameras/PanoramicCamera.cpp
    cameras/OrthographicCamera.h
    cameras/OrthographicCamera.cpp
    cameras/CompoundEye.h
    cameras/CompoundEye.cpp
    cameras/DataRecordCamera.h
    shaders.cu
    )



include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CUDA_NVRTC_ENABLED)
  CUDA_COMPILE_PTX(ptx_files ${sources})
endif()

# Make the library.
set(compoundray_target "compoundray_sdk")
add_library(${compoundray_target} ${sources})
if( WIN32 )
  target_compile_definitions( ${compoundray_target} PUBLIC GLAD_GLAPI_EXPORT )
endif()

target_link_libraries(${compoundray_target} LINK_PRIVATE
  ${GLFW_LIB_NAME}
  glad
  imgui
  ${CUDA_LIBRARIES}
  )

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${compoundray_target} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${compoundray_target} LINK_PRIVATE ${STATIC_LIBSTDCPP})
  endif()
endif()


if(CUDA_NVRTC_ENABLED)
  target_link_libraries(${compoundray_target} LINK_PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${compoundray_target} LINK_PRIVATE winmm.lib)
endif()

# Make the list of sources available to the parent directory for installation needs.
set(compoundray_sources "${sources}" PARENT_SCOPE)

set_property(TARGET ${compoundray_target} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

# Disable until we get binary samples
if(0 AND RELEASE_INSTALL_BINARY_SAMPLES AND NOT RELEASE_STATIC_BUILD)
  # If performing a release install, we want to use rpath for our install name.
  # The executables' rpaths will then be set to @executable_path so we can invoke
  # the samples from an arbitrary location and it will still find this library.
  set_target_properties(${compoundray_target} PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    )
  install(TARGETS ${compoundray_target}
    RUNTIME DESTINATION ${SDK_BINARY_INSTALL_DIR}
    LIBRARY DESTINATION ${SDK_BINARY_INSTALL_DIR}
    )
endif()
