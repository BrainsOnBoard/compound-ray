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

#include <DemandTextureManager.h>
#include <sutil/Exception.h>

#include <cassert>
#include <cstring>

const unsigned int NUM_PAGES            = 1024 * 1024;  // 1M 64KB pages => 64 GB (virtual)
const unsigned int MAX_REQUESTED_PAGES  = 1024;
const unsigned int MAX_NUM_FILLED_PAGES = 1024;

const unsigned int MAX_NUM_DEVICES = 8;

// Construct demand texture manager, initializing the OptiX paging library.
DemandTextureManager::DemandTextureManager( const std::vector<unsigned int>& devices )
{
    m_perDeviceStates.resize( MAX_NUM_DEVICES );
    for( unsigned int currDevice : devices )
    {
        PerDeviceState& currState = m_perDeviceStates[currDevice];
        currState.isActive        = true;

        CUDA_CHECK( cudaSetDevice( currDevice ) );

        // Configure the paging library.
        OptixPagingOptions options{NUM_PAGES, NUM_PAGES};
        optixPagingCreate( &options, &currState.pagingContext );
        OptixPagingSizes sizes{};
        optixPagingCalculateSizes( options.initialVaSizeInPages, sizes );

        // Allocate device memory required by the paging library.
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &currState.pagingContext->pageTable ), sizes.pageTableSizeInBytes ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &currState.pagingContext->usageBits ), sizes.usageBitsSizeInBytes ) );
        optixPagingSetup( currState.pagingContext, sizes, 1 );

        // Allocate device memory that is used to call paging library routines.
        // These allocations are retained to reduce allocation overhead.
        CUDA_CHECK( cudaMalloc( &currState.devRequestedPages, MAX_REQUESTED_PAGES * sizeof( uint32_t ) ) );
        CUDA_CHECK( cudaMalloc( &currState.devNumPagesReturned, 3 * sizeof( uint32_t ) ) );
        CUDA_CHECK( cudaMalloc( &currState.devFilledPages, MAX_NUM_FILLED_PAGES * sizeof( MapType ) ) );
    }
}

DemandTextureManager::~DemandTextureManager()
{
    try
    {
        for( PerDeviceState& state : m_perDeviceStates )
        {
            if( state.isActive )
            {
                // Free device memory and destroy the paging system.
                CUDA_CHECK( cudaFree( state.pagingContext->pageTable ) );
                CUDA_CHECK( cudaFree( state.pagingContext->usageBits ) );
                optixPagingDestroy( state.pagingContext );

                CUDA_CHECK( cudaFree( state.devRequestedPages ) );
                CUDA_CHECK( cudaFree( state.devNumPagesReturned ) );
                CUDA_CHECK( cudaFree( state.devFilledPages ) );
            }
        }
    }
    catch( ... )
    {
    }
}

// Extract texture id from page id.
static unsigned int getTextureId( uint32_t pageId )
{
    return pageId >> 4;
}

// Extract miplevel from page id.
static unsigned int getMipLevel( uint32_t pageId )
{
    return pageId & 0x0F;
}

// Create a demand-loaded texture with the specified dimensions and format.  The texture initially has no
// backing storage.
const DemandTexture& DemandTextureManager::createTexture( std::shared_ptr<ImageReader> imageReader )
{
    std::vector<unsigned int> usedDevices;
    for( unsigned int i = 0; i < m_perDeviceStates.size(); ++i )
    {
        if( m_perDeviceStates[i].isActive )
            usedDevices.push_back( i );
    }

    // Add new texture to the end of the list of textures.  The texture identifier is simply its
    // index in the DemandTexture array, which also serves as an index into the device-side
    // DemandTextureSampler array.  The texture holds a pointer to the image, from which miplevel
    // data is obtained on demand.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );
    m_textures.emplace_back( DemandTexture( usedDevices, textureId, imageReader ) );
    DemandTexture& texture = m_textures.back();
    texture.init();

    // Create texture sampler, which will be synched to the device in launchPrepare().  Note that we
    // don't set m_hostSamplersDirty when adding new samplers.
    for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); deviceIndex++ )
    {
        PerDeviceState& state = m_perDeviceStates[deviceIndex];
        if( state.isActive )
            state.hostSamplers.emplace_back( texture.getSampler( static_cast<unsigned int>( deviceIndex ) ) );
    }

    return texture;
}

// Prepare for launch, updating device-side demand texture samplers.
void DemandTextureManager::launchPrepare( unsigned int deviceIndex )
{
    CUDA_CHECK( cudaSetDevice( deviceIndex ) );

    PerDeviceState& state = m_perDeviceStates[deviceIndex];

    // Are there new samplers?
    size_t numOldSamplers = state.numDevSamplers;
    size_t numNewSamplers = m_textures.size() - numOldSamplers;
    if( numNewSamplers == 0 )
    {
        // No new samplers.  Sync existing texture samplers to device if they're dirty.
        if( state.hostSamplersDirty )
        {
            CUDA_CHECK( cudaMemcpy( state.devSamplers, state.hostSamplers.data(),
                                    state.hostSamplers.size() * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
            state.hostSamplersDirty = false;
        }
    }
    else
    {
        // Reallocate device sampler array.
        DemandTextureSampler* oldSamplers = state.devSamplers;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.devSamplers ), m_textures.size() * sizeof( DemandTextureSampler ) ) );

        // If any samplers are dirty (e.g. textures were reallocated), copy them all from the host.
        if( state.hostSamplersDirty )
        {
            CUDA_CHECK( cudaMemcpy( state.devSamplers, state.hostSamplers.data(),
                                    state.hostSamplers.size() * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
            state.hostSamplersDirty = false;
        }
        else
        {
            // Otherwise copy the old samplers from device memory and the new samplers from host memory.
            if( numOldSamplers > 0 )
            {
                CUDA_CHECK( cudaMemcpy( state.devSamplers, oldSamplers, numOldSamplers * sizeof( DemandTextureSampler ),
                                        cudaMemcpyDeviceToDevice ) );
            }
            CUDA_CHECK( cudaMemcpy( state.devSamplers + numOldSamplers, &(state.hostSamplers[numOldSamplers]),
                                    numNewSamplers * sizeof( DemandTextureSampler ), cudaMemcpyHostToDevice ) );
        }
        CUDA_CHECK( cudaFree( oldSamplers ) );
        state.numDevSamplers = m_textures.size();
    }
}

// Process requests for missing miplevels (from optixPagingMapOrRequest), reallocating textures
// and invoking callbacks to fill the new miplevels.
int DemandTextureManager::processRequests()
{
    std::vector<RequestInfo> pageRequests;
    for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); ++deviceIndex )
    {
        PerDeviceState& currState = m_perDeviceStates[deviceIndex];
        if( currState.isActive )
        {
            CUDA_CHECK( cudaSetDevice( static_cast<unsigned int>( deviceIndex ) ) );
            pullRequests( currState );
            copyRequests( currState, pageRequests, static_cast<unsigned int>( deviceIndex ) );
        }
    }
    return processRequestsImpl( pageRequests );
}

// Get page requests from the device (via optixPagingPullRequests).
void DemandTextureManager::pullRequests( PerDeviceState& state )
{
    // Get a list of requested page ids, along with lists of stale and evictable pages (which are
    // currently unused).
    optixPagingPullRequests( state.pagingContext, state.devRequestedPages, MAX_REQUESTED_PAGES, nullptr /*stalePages*/, 0,
            nullptr /*evictablePages*/, 0, state.devNumPagesReturned );
}

void DemandTextureManager::copyRequests( PerDeviceState& state, std::vector<RequestInfo>& pageRequests, unsigned int deviceIndex )
{
    // Get the sizes of the requsted, stale, and evictable page lists.
    uint32_t numReturned[3] = {0};
    CUDA_CHECK( cudaMemcpy( &numReturned[0], state.devNumPagesReturned, 3 * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );

    // Return early if no pages requested.
    uint32_t numRequests = numReturned[0];
    if( numRequests == 0 )
        return;

    // Copy the requested page list from this device.
    std::vector<uint32_t> requestedPages( numRequests );
    CUDA_CHECK( cudaMemcpy( requestedPages.data(), state.devRequestedPages, numRequests * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );

    pageRequests.reserve( pageRequests.size() + numRequests );
    for( uint32_t page : requestedPages )
        pageRequests.push_back( RequestInfo{page, deviceIndex} );
}

// Process requests.  Implemented as a separate method to permit testing.
int DemandTextureManager::processRequestsImpl( std::vector<RequestInfo>& pageRequests )
{
    if( pageRequests.empty() )
        return 0;

    // Sort the requests by page number.  This ensures that all the requests for a particular
    // texture are adjacent in the request list, allowing us to perform a single reallocation that
    // spans all the requested miplevels.
    std::sort( pageRequests.begin(), pageRequests.end(),
               []( const RequestInfo& a, const RequestInfo& b ) { return a.pageIndex < b.pageIndex; } );

    // Reallocate textures to accommodate newly requested miplevels.
    size_t numRequests = pageRequests.size();
    for( size_t i = 0; i < numRequests; /* nop */ )
    {
        RequestInfo  info        = pageRequests[i];
        uint32_t     pageId      = info.pageIndex;
        unsigned int textureId   = getTextureId( pageId );

        // Initialize the texture if necessary, e.g. reading image info from file header.
        DemandTexture* texture = &m_textures[textureId];
        if( !texture->init() )
            assert( false );  // TODO: handle image reader errors.

        unsigned int requestedMipLevel = getMipLevel( pageId );
        requestedMipLevel              = std::min( requestedMipLevel, texture->getInfo().numMipLevels );

        // The minimum and maximum mip levels requested across all devices, used for
        // determining which parts of the image to read from the disk.
        unsigned int minMipLevel = requestedMipLevel;
        unsigned int maxMipLevel = requestedMipLevel;

        // Also track minimum and maximum mip levels requested per device, so we can
        // transfer only the textures that an individual device requested.
        std::vector<MipLevelInfo> perDeviceRequests( m_perDeviceStates.size() );
        for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); ++deviceIndex )
        {
            if( m_perDeviceStates[deviceIndex].isActive )
            {
                perDeviceRequests[deviceIndex].minMipLevel = requestedMipLevel;
                perDeviceRequests[deviceIndex].maxMipLevel = requestedMipLevel;
            }
        }

        // Fetch the given mip level from disk.
        texture->readMipLevel( requestedMipLevel );

        // Accumulate requests for other miplevels from the same texture.
        for( ++i; i < numRequests && getTextureId( pageRequests[i].pageIndex ) == textureId; ++i )
        {
            uint32_t     pageId            = pageRequests[i].pageIndex;
            unsigned int deviceIndex       = pageRequests[i].deviceIndex;
            unsigned int requestedMipLevel = getMipLevel( pageId );

            texture->readMipLevel( requestedMipLevel );

            minMipLevel                                = std::min( minMipLevel, requestedMipLevel );
            maxMipLevel                                = std::max( maxMipLevel, requestedMipLevel );
            perDeviceRequests[deviceIndex].minMipLevel = std::min( minMipLevel, requestedMipLevel );
            perDeviceRequests[deviceIndex].maxMipLevel = std::max( maxMipLevel, requestedMipLevel );
        }

        // Update the host-side texture sampler.
        for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); ++deviceIndex )
        {
            PerDeviceState& currState = m_perDeviceStates[deviceIndex];
            if( currState.isActive )
            {
                // Reallocate the texture's backing storage to accomodate the new miplevels.
                // Existing miplevels are copied (using a device-to-device memcpy).
                texture->reallocate( static_cast<unsigned int>( deviceIndex ), perDeviceRequests[deviceIndex].minMipLevel,
                                     perDeviceRequests[deviceIndex].maxMipLevel );

                currState.hostSamplers[textureId] = texture->getSampler( static_cast<unsigned int>( deviceIndex ) );
                currState.hostSamplersDirty       = true;
            }
        }
    }

    // Fill each requested miplevel.
    std::vector<std::vector<MapType>> filledPages( m_perDeviceStates.size() );
    for( size_t i = 0; i < numRequests; ++i )
    {
        uint32_t       pageId      = pageRequests[i].pageIndex;
        uint32_t       deviceIndex = pageRequests[i].deviceIndex;
        DemandTexture& texture     = m_textures[getTextureId( pageId )];
        unsigned int   mipLevel    = getMipLevel( pageId );

        texture.fillMipLevel( deviceIndex, mipLevel );

        // Keep track of which pages were filled.  (The value of the page table entry is not used.)
        filledPages[deviceIndex].push_back( MapType{pageId, 1} );
    }

    // Remove mip level data from memory, now that it isn't needed anymore.
    for( DemandTexture& texture : m_textures )
        texture.flushMipLevels();

    // Push the new page mappings to the device.
    unsigned int totalRequestsFilled = 0;
    for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); ++deviceIndex )
    {
        PerDeviceState& currState = m_perDeviceStates[deviceIndex];
        if( currState.isActive )
        {
            uint32_t numFilledPages = static_cast<uint32_t>( filledPages[deviceIndex].size() );
            totalRequestsFilled += numFilledPages;
            CUDA_CHECK( cudaSetDevice( static_cast<unsigned int>( deviceIndex ) ) );
            CUDA_CHECK( cudaMemcpy( currState.devFilledPages, filledPages[deviceIndex].data(),
                                    numFilledPages * sizeof( MapType ), cudaMemcpyHostToDevice ) );
            optixPagingPushMappings( currState.pagingContext, currState.devFilledPages, numFilledPages,
                                     nullptr /*invalidatedPages*/, 0 );
        }
    }
    return totalRequestsFilled;
}
