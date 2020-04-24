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

#include <DemandLoading/ImageReader.h>

#include <ImfFrameBuffer.h>
#include <ImfTiledInputFile.h>

#include <memory>
#include <string>
#include <vector>

/// OpenEXR image reader.
class EXRReader : public ImageReader
{
  public:
    /// The constructor copies the given filename.  The file is not opened until open() is called.
    EXRReader( const char* filename )
        : m_filename( filename )
    {
    }

    /// Destructor
    ~EXRReader() override { close(); }

    /// Open the image and read header info, including dimensions and format.  Returns false on error.
    bool open( Info* info ) override;

    /// Close the image.
    void close() override;
    
    /// Read the specified miplevel into the given buffer.  Returns true for success.
    bool readMipLevel( float4* dest, unsigned int miplevel, unsigned int width, unsigned int height ) override;

    // Read the specified tile into the given buffer.  Returns true for success.
    bool readTile( float4* dest, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) override;

  private:
    std::string                          m_filename;
    std::unique_ptr<Imf::TiledInputFile> m_inputFile;
    Info                                 m_info;
    std::vector<float4>                  m_tileBuffer;
    std::unique_ptr<Imf::FrameBuffer>    m_tileFrameBuffer;
};
