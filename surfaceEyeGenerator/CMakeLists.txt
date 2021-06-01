OPTIX_add_sample_executable( surfaceEyeGenerator target_name
  Surface.h
  Surface.cpp

  surfaceEyeGenerator.cpp

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
