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

#include "DemandTexture.h"
#include <sutil/Exception.h>

#include <cassert>
#include <cstring>

const unsigned int MAX_NUM_DEVICES = 8;

// DemandTexture is constructed by DemandTextureManager::createTexture
DemandTexture::DemandTexture( const std::vector<unsigned int>& devices, unsigned int id, std::shared_ptr<ImageReader> image )
    : m_id( id )
    , m_image( image )
{
    m_perDeviceStates.resize( MAX_NUM_DEVICES );
    for( unsigned int device : devices )
        m_perDeviceStates[device].isActive = true;
}

DemandTexture::~DemandTexture()
{
    try
    {
        for( size_t deviceIndex = 0; deviceIndex < m_perDeviceStates.size(); deviceIndex++ )
        {
            PerDeviceState& currState = m_perDeviceStates[deviceIndex];
            if( currState.isActive )
            {
                CUDA_CHECK( cudaDestroyTextureObject( currState.texture ) );
                CUDA_CHECK( cudaFreeMipmappedArray( currState.mipLevelData ) );
            }
        }
    }
    catch( ... )
    {
    }
}

// Initialize the texture, e.g. reading image info from file header.  Returns false on error.
bool DemandTexture::init()
{
    if( !m_isInitialized )
    {
        m_isInitialized = true;
        return m_image->open( &m_info );
    }
    return true;
}

// Reallocate backing storage to span the specified miplevels.
void DemandTexture::reallocate( unsigned int deviceIndex, unsigned int minMipLevel, unsigned int maxMipLevel )
{
    PerDeviceState& currState = m_perDeviceStates[deviceIndex];

    unsigned int oldMinMipLevel = currState.minMipLevel;
    unsigned int oldMaxMipLevel = currState.maxMipLevel;
    if( minMipLevel >= oldMinMipLevel && maxMipLevel <= oldMaxMipLevel )
        return;

    unsigned newMinMipLevel = std::min( oldMinMipLevel, minMipLevel );
    unsigned newMaxMipLevel = std::max( oldMaxMipLevel, maxMipLevel );
    currState.minMipLevel   = newMinMipLevel;
    currState.maxMipLevel   = newMaxMipLevel;

    unsigned int newWidth  = getLevelWidth( newMinMipLevel );
    unsigned int newHeight = getLevelHeight( newMinMipLevel );
    unsigned int numLevels = newMaxMipLevel - newMinMipLevel + 1;

    CUDA_CHECK( cudaSetDevice( deviceIndex ) );

    // Allocate new array.
    cudaMipmappedArray_t  newMipLevelData;
    const cudaChannelFormatDesc& channelDesc = getInfo().channelDesc;
    cudaExtent                   extent      = make_cudaExtent( newWidth, newHeight, 0 );
    CUDA_CHECK( cudaMallocMipmappedArray( &newMipLevelData, &channelDesc, extent, numLevels ) );

    // Copy any existing levels from the old array.
    cudaMipmappedArray_t oldMipLevelData = currState.mipLevelData;
    currState.mipLevelData               = newMipLevelData;
    for( unsigned int nominalLevel = oldMinMipLevel; nominalLevel <= oldMaxMipLevel; ++nominalLevel )
    {
        unsigned int sourceLevel  = nominalLevel - oldMinMipLevel;
        unsigned int destLevel    = nominalLevel - newMinMipLevel;
        unsigned int width        = getLevelWidth( nominalLevel );
        unsigned int height       = getLevelHeight( nominalLevel );
        unsigned int widthInBytes = width * (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8;

        // Get the CUDA arrays for the source and destination miplevels.
        cudaArray_t sourceArray, destArray;
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &sourceArray, oldMipLevelData, sourceLevel ) );
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &destArray, newMipLevelData, destLevel ) );

        // Copy the miplevel.
        CUDA_CHECK( cudaMemcpy2DArrayToArray( destArray, 0, 0, sourceArray, 0, 0, widthInBytes, height, cudaMemcpyDeviceToDevice ) );
    }

    // Destroy the old mipmapped array and the old texture.
    CUDA_CHECK( cudaFreeMipmappedArray( oldMipLevelData ) );
    CUDA_CHECK( cudaDestroyTextureObject( currState.texture ) );

    // Create new texture object.
    currState.texture = createTextureObject( deviceIndex );
}

// Create CUDA texture object (called internally after reallocation).
cudaTextureObject_t DemandTexture::createTextureObject( unsigned int deviceIndex ) const
{
    const PerDeviceState& currState = m_perDeviceStates[deviceIndex];

    // Create resource description
    cudaResourceDesc resDesc  = {};
    resDesc.resType           = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = currState.mipLevelData;

    // Construct texture description with various options.
    cudaTextureDesc texDesc     = {};
    texDesc.addressMode[0]      = cudaAddressModeWrap;
    texDesc.addressMode[1]      = cudaAddressModeWrap;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.maxMipmapLevelClamp = static_cast<float>( currState.maxMipLevel );
    texDesc.minMipmapLevelClamp = 0.f;
    texDesc.mipmapFilterMode    = cudaFilterModeLinear;
    texDesc.normalizedCoords    = 1;
    texDesc.readMode            = cudaReadModeElementType;
    texDesc.maxAnisotropy       = MAX_ANISOTROPY;

    // Bias miplevel access in demand loaded texture based on the current minimum miplevel loaded.
    texDesc.mipmapLevelBias = -static_cast<float>( currState.minMipLevel );

    // Create texture object
    cudaTextureObject_t texture;
    CUDA_CHECK( cudaCreateTextureObject( &texture, &resDesc, &texDesc, nullptr /*cudaResourceViewDesc*/ ) );
    return texture;
}

void DemandTexture::readMipLevel( unsigned int nominalMipLevel )
{
    unsigned int width  = getLevelWidth( nominalMipLevel );
    unsigned int height = getLevelHeight( nominalMipLevel );
    if( m_mipLevels.size() <= nominalMipLevel )
        m_mipLevels.resize( nominalMipLevel + 1 );
    m_mipLevels[nominalMipLevel].resize( width * height );
    if( !m_image->readMipLevel( m_mipLevels[nominalMipLevel].data(), nominalMipLevel, width, height ) )
        assert( false );  // TODO: handle image read failure
}

// Fill the specified miplevel.
void DemandTexture::fillMipLevel( unsigned int deviceIndex, unsigned int nominalMipLevel )
{
    PerDeviceState& currState = m_perDeviceStates[deviceIndex];

    CUDA_CHECK( cudaSetDevice( deviceIndex ) );

    // Get the backing storage for the specified miplevel.
    assert( currState.minMipLevel <= nominalMipLevel
            && nominalMipLevel <= currState.maxMipLevel );
    unsigned int actualMipLevel = nominalMipLevel - currState.minMipLevel;
    cudaArray_t  array;
    CUDA_CHECK( cudaGetMipmappedArrayLevel( &array, currState.mipLevelData, actualMipLevel ) );

    unsigned int width  = getLevelWidth( nominalMipLevel );
    unsigned int height = getLevelHeight( nominalMipLevel );
    size_t widthInBytes = width * sizeof( float4 );
    size_t pitch        = widthInBytes;
    CUDA_CHECK( cudaMemcpy2DToArray( array, 0, 0, m_mipLevels[nominalMipLevel].data(), pitch, widthInBytes, height, cudaMemcpyHostToDevice ) );
}

void DemandTexture::flushMipLevels()
{
    for( std::vector<float4>& mipLevel : m_mipLevels )
        mipLevel.resize( 0 );
}
