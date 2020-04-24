#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

set( OPTIX_PAGING "${OptiX_INCLUDE}/optixPaging" )
set( EXP_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib" )
include_directories(
  .
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${EXP_LIB_DIR}"
  "${OPTIX_PAGING}"
)

# These CUDA files need to be compiled as OBJ, not as PTX
set_source_files_properties( ${OPTIX_PAGING}/optixPaging.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

# This file is just for reference in the IDE, don't compile it.
set_source_files_properties( ${OPTIX_PAGING}/optixPagingImpl.cpp PROPERTIES HEADER_FILE_ONLY TRUE )

OPTIX_add_sample_executable( optixDemandTexture target_name
  optixDemandTexture.cpp
  optixDemandTexture.cu
  optixDemandTexture.h
  DemandTexture.cpp
  DemandTexture.h
  DemandTextureManager.cpp
  DemandTextureManager.h
  ${EXP_LIB_DIR}/DemandLoading/ImageReader.h
  ${EXP_LIB_DIR}/DemandLoading/CheckerBoardReader.h
  ${EXP_LIB_DIR}/DemandLoading/CheckerBoardReader.cpp
  ${OPTIX_PAGING}/optixPaging.h
  ${OPTIX_PAGING}/optixPaging.cu
  # EXRReader.cpp is added below if OpenEXR is available.
  )

find_package( OpenEXR )
if( NOT OpenEXR_FOUND )
  message( WARNING "OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

find_package( ZlibStatic )
if ( NOT ZlibStatic_FOUND )
  message( WARNING "Zlib not found (see ZlibStatic_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

if( OpenEXR_FOUND AND ZlibStatic_FOUND )
  # Build our EXRImage class if OpenEXR is available, adding the necessary include path and libraries.
  target_sources( ${target_name} PUBLIC
    ${EXP_LIB_DIR}/DemandLoading/EXRReader.cpp
    ${EXP_LIB_DIR}/DemandLoading/EXRReader.h )
  target_compile_definitions( ${target_name} PUBLIC OPTIX_SAMPLE_USE_OPEN_EXR )

  target_include_directories( ${target_name} PUBLIC "${OpenEXR_INCLUDE_DIR}" )
  target_link_libraries( ${target_name} ${OpenEXR_LIBRARIES} ${ZlibStatic_LIBRARIES} ${CUDA_LIBRARIES} )
else()
  target_link_libraries( ${target_name} ${CUDA_LIBRARIES} )
endif()
