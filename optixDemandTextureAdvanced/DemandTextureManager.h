//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "DemandTexture.h"
#include <optixPaging/optixPaging.h>

#include <memory>
#include <vector>

/// DemandTextureManager demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandTextureManager
{
  public:
    /// Construct demand texture manager, initializing the OptiX paging library.
    DemandTextureManager( const std::vector<unsigned int>& devices );

    /// Destroy demand texture manager, reclaiming host and device memory.
    ~DemandTextureManager();

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The read() method is invoked on the image to fill each require miplevel.
    /// The image pointer is retained indefinitely.
    const DemandTexture& createTexture( std::shared_ptr<ImageReader> image );

    /// Prepare for launch, updating device-side demand texture samplers.
    void launchPrepare( unsigned int deviceIndex );

    /// Process requests for missing miplevels (from optixPagingMapOrRequest), reallocating textures
    /// and invoking callbacks to fill the new miplevels.
    int processRequests();

    /// Get the OptiX paging library context, which is passed as a launch parameter and used to call
    /// optixPagingMapOrRequest.
    const OptixPagingContext& getPagingContext( unsigned int deviceIndex ) const
    {
        return *( m_perDeviceStates[deviceIndex].pagingContext );
    }

    /// Get the array of device-side demand texture samplers, which is indexed by texture id.
    const DemandTextureSampler* getSamplers( unsigned int deviceIndex ) { return m_perDeviceStates[deviceIndex].devSamplers; }

  private:
    // Vector of demand-loaded textures (owned by the DemandTextureManager), indexed by texture id.
    std::vector<DemandTexture> m_textures;

    struct PerDeviceState
    {
        bool isActive = false;

        // The OptiX paging system employs a context that includes the page table, etc.
        OptixPagingContext* pagingContext = nullptr;

        // Host and device arrays of texture samplers.
        std::vector<DemandTextureSampler> hostSamplers;
        bool                              hostSamplersDirty = false;

        DemandTextureSampler* devSamplers    = nullptr;
        size_t                numDevSamplers = 0;

        // Device memory used to call OptiX paging library routines.
        // These allocations are retained to reduce allocation overhead.
        uint32_t* devRequestedPages   = nullptr;
        uint32_t* devNumPagesReturned = nullptr;
        MapType*  devFilledPages      = nullptr;

        // The minimum and maximum mip levels requested by this device.
        unsigned int minMipLevel = 0;
        unsigned int maxMipLevel = 0;
    };
    std::vector<PerDeviceState> m_perDeviceStates;

    struct RequestInfo
    {
        uint32_t     pageIndex;
        unsigned int deviceIndex;
    };

    struct MipLevelInfo
    {
        unsigned int minMipLevel = 0;
        unsigned int maxMipLevel = 0;
    };

    // Get page requests from the device (via optixPagingPullRequests).
    void pullRequests( PerDeviceState& state );

    void copyRequests( PerDeviceState& deviceState, std::vector<RequestInfo>& pageRequests, unsigned int deviceIndex );

    // Process requests.  Implemented as a separate method to permit testing.
    int processRequestsImpl( std::vector<RequestInfo>& pageRequests );
};
