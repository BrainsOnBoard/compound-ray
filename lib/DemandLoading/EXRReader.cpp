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

#include <DemandLoading/EXRReader.h>
#include <sutil/Exception.h>

#include <cuda_runtime.h>

#include <ImfChannelList.h>
#include <ImfTiledInputFile.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

using namespace Imf;
using namespace Imath;

// Open the image and read header info, including dimensions and format.  Returns false on error.
bool EXRReader::open( Info* info )
{
    // Open input file.
    // TODO: handle file not found, etc.
    assert( !m_inputFile );
    m_inputFile.reset( new TiledInputFile( m_filename.c_str() ) );

    // Get the width and height from the data window of the finest miplevel.
    Box2i dw           = m_inputFile->dataWindowForLevel( 0, 0 );
    m_info.width        = dw.max.x - dw.min.x + 1;
    m_info.height       = dw.max.y - dw.min.y + 1;
    m_info.numMipLevels = m_inputFile->numLevels();

    // Use float format.  Note that half and integer channels will be converted to float by the
    // FrameBuffer/Slice logic below.  We fill in the struct ourselves rather than calling
    // cudaCreateChannelDesc to avoid a CUDA API dependency here.
    const int floatBits = sizeof( float ) * 8;
    m_info.channelDesc   = cudaChannelFormatDesc{floatBits, floatBits, floatBits, floatBits, cudaChannelFormatKindFloat};

    // Allocate buffer for reading tile data.
    m_info.tileWidth  = m_inputFile->tileXSize();
    m_info.tileHeight = m_inputFile->tileYSize();
    m_tileBuffer.resize( m_info.tileWidth * m_info.tileHeight );

    *info = m_info;
    return true;
}

// Close the image.
void EXRReader::close()
{
    m_inputFile.reset();
}

// Read all the tiles for the specified miplevel.  Returns true for success.
bool EXRReader::readMipLevel( float4* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight )
{
    assert( m_inputFile );

    // Get data window offset and dimensions.
    Box2i dw     = m_inputFile->dataWindowForLevel( mipLevel, mipLevel );
    int   dx     = dw.min.x;
    int   dy     = dw.min.y;
    int   width  = dw.max.x - dx + 1;
    assert( width == static_cast<int>( expectedWidth ) && dw.max.y - dy + 1 == static_cast<int>( expectedHeight ) );

    // Set up FrameBuffer, which is an adapter for the pixel array.  We must
    // provide a base pointer that is offset to account for the data window.
    float4*     base    = dest - dx - dy * width;
    size_t      xStride = sizeof( float4 );
    size_t      yStride = width * sizeof( float4 );
    FrameBuffer frameBuffer;
    frameBuffer.insert( "R", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->x ), xStride, yStride ) );
    frameBuffer.insert( "G", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->y ), xStride, yStride ) );
    frameBuffer.insert( "B", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->z ), xStride, yStride ) );
    frameBuffer.insert( "A", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->w ), xStride, yStride ) );
    m_inputFile->setFrameBuffer( frameBuffer );

    // Read all the tiles for the specified miplevel.
    m_inputFile->readTiles( 0, m_inputFile->numXTiles( mipLevel ) - 1, 0, m_inputFile->numYTiles( mipLevel ) - 1, mipLevel, mipLevel );

    return true;
}

// Read the specified tile.  Returns true for success.
bool EXRReader::readTile( float4* dest, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    assert( m_inputFile );
    assert( m_tileBuffer.size() >= m_info.tileWidth * m_info.tileHeight );

    Box2i dw = m_inputFile->dataWindowForTile( tileX, tileY, mipLevel );
    assert(dw.max.x - dw.min.x + 1 == static_cast<int>(m_info.tileWidth));
    assert(dw.max.y - dw.min.y + 1 == static_cast<int>(m_info.tileHeight));

    // Set up FrameBuffer, which is an adapter for the pixel array.  We must provide a base pointer
    // that is offset to account for the data window.  TODO: how to reuse FrameBuffer?  It's
    // expensive to construct and set.  But the data window is different for every tile?
    m_tileFrameBuffer.reset( new FrameBuffer );
    size_t  xStride = sizeof( float4 );
    size_t  yStride = m_info.tileWidth * sizeof( float4 );
    float4* base = dest - dw.min.x - dw.min.y * m_info.tileWidth;

    m_tileFrameBuffer->insert( "R", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->x ), xStride, yStride ) );
    m_tileFrameBuffer->insert( "G", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->y ), xStride, yStride ) );
    m_tileFrameBuffer->insert( "B", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->z ), xStride, yStride ) );
    m_tileFrameBuffer->insert( "A", Slice( PixelType::FLOAT, reinterpret_cast<char*>( &base->w ), xStride, yStride ) );
    m_inputFile->setFrameBuffer( *m_tileFrameBuffer );

    // Read tile data into m_tileBuffer via the FrameBuffer created above.
    m_inputFile->readTile( tileX, tileY, mipLevel );
    return m_tileBuffer.data();
}
