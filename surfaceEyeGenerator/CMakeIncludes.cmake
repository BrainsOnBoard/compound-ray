
macro(add_eye_generator relative_path sources)
  include_directories(${relative_path})
  set(eye_generator_sources
      #${relative_path}/Test.cpp
      ${relative_path}/main.cpp
      ${relative_path}/EyeGenerator.cpp
      ${relative_path}/EyeGenerator.h
      ${relative_path}/EquilibriumGenerator.cpp
      ${relative_path}/EquilibriumGenerator.h
      ${relative_path}/NonPlanarCoordinate.cpp
      ${relative_path}/NonPlanarCoordinate.h
      ${relative_path}/SphericalCoordinate.cpp
      ${relative_path}/SphericalCoordinate.h
      ${relative_path}/SinewaveDropletCoordinate.cpp
      ${relative_path}/SinewaveDropletCoordinate.h
     )
  list(APPEND ${sources} ${eye_generator_sources})
endmacro()

