OPTIX_add_sample_executable( surfaceEyeGenerator target_name
  surfaceEyeGenerator.cpp

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
