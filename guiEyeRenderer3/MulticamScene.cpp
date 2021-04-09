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
//        This file was originally based on the "Scene.cpp" file 
//        that comes within sutil of the NVidia OptiX SDK, but has
//        been changed by Blayze Millward to be more aligned to the
//        design schema of the insect eye perspective renderer.

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>

#include "MulticamScene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace
{

float3 make_float3_from_double( double x, double y, double z )
{
    return make_float3( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ) );
}

float4 make_float4_from_double( double x, double y, double z, double w )
{
    return make_float4( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ), static_cast<float>( w ) );
}

typedef Record<globalParameters::HitGroupData> HitGroupRecord;

void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

template<typename T>
BufferView<T> bufferViewFromGLTF( const tinygltf::Model& model, MulticamScene& scene, const int32_t accessor_idx )
{
    if( accessor_idx == -1 )
        return BufferView<T>();

    const auto& gltf_accessor    = model.accessors[ accessor_idx ];
    const auto& gltf_buffer_view = model.bufferViews[ gltf_accessor.bufferView ];

    const int32_t elmt_byte_size =
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2 :
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT   ? 4 :
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT          ? 4 :
            0;
    if( !elmt_byte_size )
        throw Exception( "gltf accessor component type not supported" );

    const CUdeviceptr buffer_base = scene.getBuffer( gltf_buffer_view.buffer );
    BufferView<T> buffer_view;
    buffer_view.data           = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride    = static_cast<uint16_t>( gltf_buffer_view.byteStride );
    buffer_view.count          = static_cast<uint32_t>( gltf_accessor.count );
    buffer_view.elmt_byte_size = static_cast<uint16_t>( elmt_byte_size );

    return buffer_view;
}

const bool isObjectsExtraValueTrue (const tinygltf::Value& extras, const char* key)
{
  tinygltf::Value v = extras.Get(key);
  if(v.IsBool())
  {
    return v.Get<bool>();
  }

  if(v.IsString())
  {
     std::string valueStr = v.Get<std::string>();
     std::transform(valueStr.begin(), valueStr.end(), valueStr.begin(), [](unsigned char c){ return std::tolower(c); });
     return (valueStr.compare("true") == 0);
  }
  return false;
}
const std::vector<std::string> splitString(const std::string& s, const std::string& deliminator)
{
  std::vector<std::string> output;
  const size_t delimSize = deliminator.size();
  size_t lastDelimLoc = 0;
  size_t delimLoc = s.find(deliminator, 0);
  while(delimLoc != std::string::npos)
  {
    if(delimLoc != lastDelimLoc)
      output.push_back(s.substr(lastDelimLoc, delimLoc-lastDelimLoc));
    lastDelimLoc = delimLoc + delimSize;
    delimLoc = s.find(deliminator, lastDelimLoc);
  }
  // Push either the whole thing if it's not found, or the last segment if there were deliminators
  output.push_back(s.substr(lastDelimLoc, s.size()));
  return output;
}

void processGLTFNode(
        MulticamScene& scene,
        const tinygltf::Model& model,
        const tinygltf::Node& gltf_node,
        const Matrix4x4& parent_matrix
        )
{
    const Matrix4x4 translation = gltf_node.translation.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::translate( make_float3_from_double(
                    gltf_node.translation[0],
                    gltf_node.translation[1],
                    gltf_node.translation[2]
                    ) );

    const Matrix4x4 rotation = gltf_node.rotation.empty() ?
        Matrix4x4::identity() :
        Quaternion(
                static_cast<float>( gltf_node.rotation[3] ),
                static_cast<float>( gltf_node.rotation[0] ),
                static_cast<float>( gltf_node.rotation[1] ),
                static_cast<float>( gltf_node.rotation[2] )
                ).rotationMatrix();

    const Matrix4x4 scale = gltf_node.scale.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::scale( make_float3_from_double(
                    gltf_node.scale[0],
                    gltf_node.scale[1],
                    gltf_node.scale[2]
                    ) );

    std::vector<float> gltf_matrix;
    for( double x : gltf_node.matrix )
        gltf_matrix.push_back( static_cast<float>( x ) );
    const Matrix4x4 matrix = gltf_node.matrix.empty() ?
        Matrix4x4::identity() :
        Matrix4x4( reinterpret_cast<float*>( gltf_matrix.data() ) ).transpose();

    const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale ;

    if( gltf_node.camera != -1 )
    {
        const auto& gltf_camera = model.cameras[ gltf_node.camera ];
        std::cerr << "============================"<<std::endl<<"Processing camera '" << gltf_camera.name << "'" << std::endl
            << "\ttype: " << gltf_camera.type << std::endl;

        // Get configured camera information and local axis
        const float3 upAxis      = make_float3( node_xform*make_float4_from_double( 0.0f, 1.0f,  0.0f, 0.0f ) );
        const float3 forwardAxis = make_float3( node_xform*make_float4_from_double( 0.0f, 0.0f, -1.0f, 0.0f ) );
        const float3 rightAxis   = make_float3( node_xform*make_float4_from_double( 1.0f, 0.0f,  0.0f, 0.0f ) );

        const float3 eye     = make_float3( node_xform*make_float4_from_double( 0.0f, 0.0f,  0.0f, 1.0f ) );
        const float  yfov   = static_cast<float>( gltf_camera.perspective.yfov ) * 180.0f / static_cast<float>( M_PI );
        std::cerr << "\teye   : " << eye.x    << ", " << eye.y    << ", " << eye.z    << std::endl;
        std::cerr << "\tfov   : " << yfov     << std::endl;
        std::cerr << "\taspect: " << gltf_camera.perspective.aspectRatio << std::endl;

        // Form camera objects
        if( gltf_camera.type == "orthographic" )
        {
          std::cerr << "Adding orthographic camera..."<<std::endl;
          OrthographicCamera* camera = new OrthographicCamera(gltf_camera.name);
          camera->setPosition(eye);
          camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
          camera->setXYscale(gltf_camera.orthographic.xmag, gltf_camera.orthographic.ymag);
          scene.addCamera(camera);
          return;
        }

        if(isObjectsExtraValueTrue(gltf_camera.extras, "panoramic"))
        {
          std::cerr << "This camera has special indicator 'panoramic' specified, adding panoramic camera..."<<std::endl;
          PanoramicCamera* camera = new PanoramicCamera(gltf_camera.name);
          // TODO
          //if(gltf_camera.extras.Has("near-clip-radius"))// TODO: Define the near-clip-radius in panoramic camera class as static
          //{
          //  float nearClipRadius = std::stof(gltf_camera.extras.Get("near-clip-radius"));
          //  std::cerr << "This camera has special indicator 'panoramic' specified, adding panoramic camera..."<<std::endl;
          //  camera->setStartRadius(nearClipRadius);
          //}
          camera->setPosition(eye);
          camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
          scene.addCamera(camera);
          return;
        }

        if(isObjectsExtraValueTrue(gltf_camera.extras, "compound-eye"))
        {
          std::cerr << "This camera has special indicator 'insect-eye' specified, adding compound eye based camera..."<<std::endl;
          std::string eyeDataPath = gltf_camera.extras.Get("compound-structure").Get<std::string>();
          std::string projectionShader = gltf_camera.extras.Get("compound-projection").Get<std::string>();
          std::cerr << "  Camera internal projection type: "<<projectionShader<<std::endl;
          std::cerr << "  Camera eye data path: "<<eyeDataPath<<std::endl;
          // Load the file
          std::ifstream eyeDataFile(eyeDataPath);
          if(eyeDataFile.is_open())
          {
            // Read the lines of the file
            std::string line;
            std::vector<Ommatidium> ommVector;// Stores the ommatidia
            size_t ommCount = 0;
            while(std::getline(eyeDataFile, line))
            {
              std::vector<std::string> splitData = splitString(line, " ");// position, direction, angle
              Ommatidium o = {{std::stof(splitData[0]), std::stof(splitData[1]), std::stof(splitData[2])}, {std::stof(splitData[3]), std::stof(splitData[4]), std::stof(splitData[5])}, std::stof(splitData[6]) };
              ommVector.push_back(o);
              ommCount++;
            }
            std::cerr<< "  > Loaded "<<ommCount<<" ommatidia."<<std::endl;

            // Create a new compound eye
            CompoundEye* camera = new CompoundEye(gltf_camera.name, projectionShader, ommVector.size());
            camera->setPosition(eye);
            camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
            scene.addCamera(camera);
            camera->copyOmmatidia(ommVector.data());
            camera->setCompoundIndex(scene.addCompoundCamera(camera));
          }else{
            std::cerr << "ERROR: Unable to open \"" << eyeDataPath << "\""<<std::endl;
          }
          return;
        }

        std::cout << " ACTUAL RETURN     : "<<isObjectsExtraValueTrue(gltf_camera.extras, "insecteye")<<std::endl;
        std::cerr << "Adding perspective camera..." << std::endl;

        PerspectiveCamera* camera = new PerspectiveCamera(gltf_camera.name);
        camera->setPosition(eye);
        camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
        camera->setYFOV(yfov);
        scene.addCamera( camera );
    }
    else if( gltf_node.mesh != -1 )
    {
        const auto& gltf_mesh = model.meshes[ gltf_node.mesh ];
        std::cerr << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
        std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;
        for( auto& gltf_primitive : gltf_mesh.primitives )
        {
            if( gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES ) // Ignore non-triangle meshes
            {
                std::cerr << "\tNon-triangle primitive: skipping\n";
                continue;
            }

            auto mesh = std::make_shared<MulticamScene::MeshGroup>();
            scene.addMesh( mesh );


            mesh->name = gltf_mesh.name;
            mesh->indices.push_back( bufferViewFromGLTF<uint32_t>( model, scene, gltf_primitive.indices ) );
            mesh->material_idx.push_back( gltf_primitive.material );
            mesh->transform = node_xform;
            std::cerr << "\t\tNum triangles: " << mesh->indices.back().count / 3 << std::endl;

            assert( gltf_primitive.attributes.find( "POSITION" ) !=  gltf_primitive.attributes.end() );
            const int32_t pos_accessor_idx =  gltf_primitive.attributes.at( "POSITION" );
            mesh->positions.push_back( bufferViewFromGLTF<float3>( model, scene, pos_accessor_idx ) );

            const auto& pos_gltf_accessor = model.accessors[ pos_accessor_idx ];
            mesh->object_aabb = Aabb(
                    make_float3_from_double(
                        pos_gltf_accessor.minValues[0],
                        pos_gltf_accessor.minValues[1],
                        pos_gltf_accessor.minValues[2]
                        ),
                    make_float3_from_double(
                        pos_gltf_accessor.maxValues[0],
                        pos_gltf_accessor.maxValues[1],
                        pos_gltf_accessor.maxValues[2]
                        ) );
            mesh->world_aabb = mesh->object_aabb;
            mesh->world_aabb.transform( node_xform );

            auto normal_accessor_iter = gltf_primitive.attributes.find( "NORMAL" ) ;
            if( normal_accessor_iter  !=  gltf_primitive.attributes.end() )
            {
                std::cerr << "\t\tHas vertex normals: true\n";
                mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, normal_accessor_iter->second ) );
            }
            else
            {
                std::cerr << "\t\tHas vertex normals: false\n";
                mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, -1 ) );
            }

            auto texcoord_accessor_iter = gltf_primitive.attributes.find( "TEXCOORD_0" ) ;
            if( texcoord_accessor_iter  !=  gltf_primitive.attributes.end() )
            {
                std::cerr << "\t\tHas texcoords: true\n";
                mesh->texcoords.push_back( bufferViewFromGLTF<float2>( model, scene, texcoord_accessor_iter->second ) );
            }
            else
            {
                std::cerr << "\t\tHas texcoords: false\n";
                mesh->texcoords.push_back( bufferViewFromGLTF<float2>( model, scene, -1 ) );
            }
        }
    }
    else if( !gltf_node.children.empty() )
    {
        for( int32_t child : gltf_node.children )
        {
            processGLTFNode( scene, model, model.nodes[child], node_xform );
        }
    }
}

} // end anon namespace


void loadScene( const std::string& filename, MulticamScene& scene )
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile( &model, &err, &warn, filename );
    if( !warn.empty() )
        std::cerr << "glTF WARNING: " << warn << std::endl;
    if( !ret )
    {
        std::cerr << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        throw Exception( err.c_str() );
    }

    //
    // Process buffer data first -- buffer views will reference this list
    //
    for( const auto& gltf_buffer : model.buffers )
    {
        const uint64_t buf_size = gltf_buffer.data.size();
        std::cerr << "Processing glTF buffer '" << gltf_buffer.name << "'\n"
                  << "\tbyte size: " << buf_size << "\n"
                  << "\turi      : " << gltf_buffer.uri << std::endl;

        scene.addBuffer( buf_size,  gltf_buffer.data.data() );
    }

    //
    // Images -- just load all up front for simplicity
    //
    for( const auto& gltf_image : model.images )
    {
        std::cerr << "Processing image '" << gltf_image.name << "'\n"
                  << "\t(" << gltf_image.width << "x" << gltf_image.height << ")x" << gltf_image.component << "\n"
                  << "\tbits: " << gltf_image.bits << std::endl;

        assert( gltf_image.component == 4 );
        assert( gltf_image.bits      == 8 || gltf_image.bits == 16 );

        scene.addImage(
                gltf_image.width,
                gltf_image.height,
                gltf_image.bits,
                gltf_image.component,
                gltf_image.image.data()
                );
    }

    //
    // Textures -- refer to previously loaded images
    //
    for( const auto& gltf_texture : model.textures )
    {
        if( gltf_texture.sampler == -1 )
        {
            scene.addSampler( cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source );
            continue;
        }

        const auto& gltf_sampler = model.samplers[ gltf_texture.sampler ];

        const cudaTextureAddressMode address_s = gltf_sampler.wrapS == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                                 gltf_sampler.wrapS == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                                            cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = gltf_sampler.wrapT == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                                 gltf_sampler.wrapT == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                                            cudaAddressModeWrap;
        const cudaTextureFilterMode  filter    = gltf_sampler.minFilter == GL_NEAREST     ? cudaFilterModePoint   :
                                                                                            cudaFilterModeLinear;
        scene.addSampler( address_s, address_t, filter, gltf_texture.source );
    }

    //
    // Materials
    //
    for( auto& gltf_material : model.materials )
    {
        std::cerr << "Processing glTF material: '" << gltf_material.name << "'\n";
        MaterialData::Pbr mtl;

        {
            const auto base_color_it = gltf_material.values.find( "baseColorFactor" );
            if( base_color_it != gltf_material.values.end() )
            {
                const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
                mtl.base_color = make_float4_from_double( c[0], c[1], c[2], c[3] );
                std::cerr
                    << "\tBase color: ("
                    << mtl.base_color.x << ", "
                    << mtl.base_color.y << ", "
                    << mtl.base_color.z << ")\n";
            }
            else
            {
                std::cerr << "\tUsing default base color factor\n";
            }
        }

        {
            const auto base_color_it = gltf_material.values.find( "baseColorTexture" );
            if( base_color_it != gltf_material.values.end() )
            {
                std::cerr << "\tFound base color tex: " << base_color_it->second.TextureIndex() << "\n";
                mtl.base_color_tex = scene.getSampler( base_color_it->second.TextureIndex() );
            }
            else
            {
                std::cerr << "\tNo base color tex\n";
            }
        }

        {
            const auto roughness_it = gltf_material.values.find( "roughnessFactor" );
            if( roughness_it != gltf_material.values.end() )
            {
                mtl.roughness = static_cast<float>( roughness_it->second.Factor() );
                std::cerr << "\tRougness:  " << mtl.roughness <<  "\n";
            }
            else
            {
                std::cerr << "\tUsing default roughness factor\n";
            }
        }

        {
            const auto metallic_it = gltf_material.values.find( "metallicFactor" );
            if( metallic_it != gltf_material.values.end() )
            {
                mtl.metallic = static_cast<float>( metallic_it->second.Factor() );
                std::cerr << "\tMetallic:  " << mtl.metallic <<  "\n";
            }
            else
            {
                std::cerr << "\tUsing default metallic factor\n";
            }
        }

        {
            const auto metallic_roughness_it = gltf_material.values.find( "metallicRoughnessTexture" );
            if( metallic_roughness_it != gltf_material.values.end() )
            {
                std::cerr << "\tFound metallic roughness tex: " << metallic_roughness_it->second.TextureIndex() << "\n";
                mtl.metallic_roughness_tex = scene.getSampler( metallic_roughness_it->second.TextureIndex() );
            }
            else
            {
                std::cerr << "\tNo metallic roughness tex\n";
            }
        }

        {
            const auto normal_it = gltf_material.additionalValues.find( "normalTexture" );
            if( normal_it != gltf_material.additionalValues.end() )
            {
                std::cerr << "\tFound normal color tex: " << normal_it->second.TextureIndex() << "\n";
                mtl.normal_tex = scene.getSampler( normal_it->second.TextureIndex() );
            }
            else
            {
                std::cerr << "\tNo normal tex\n";
            }
        }

        scene.addMaterial( mtl );
    }

    //
    // Process nodes
    //
    std::vector<int32_t> root_nodes( model.nodes.size(), 1 );
    for( auto& gltf_node : model.nodes )
        for( int32_t child : gltf_node.children )
            root_nodes[child] = 0;

    for( size_t i = 0; i < root_nodes.size(); ++i )
    {
        if( !root_nodes[i] )
            continue;
        auto& gltf_node = model.nodes[i];

        processGLTFNode( scene, model, gltf_node, Matrix4x4::identity() );
    }
}


void MulticamScene::addBuffer( const uint64_t buf_size, const void* data )
{
        CUdeviceptr buffer = 0;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &buffer ), buf_size ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( buffer ),
                    data,
                    buf_size,
                    cudaMemcpyHostToDevice
                    ) );
        m_buffers.push_back( buffer );
}


void MulticamScene::addImage(
                const int32_t width,
                const int32_t height,
                const int32_t bits_per_component,
                const int32_t num_components,
                const void* data
                )
{
    // Allocate CUDA array in device memory
    int32_t               pitch;
    cudaChannelFormatDesc channel_desc;
    if( bits_per_component == 8 )
    {
        pitch        = width*num_components*sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if( bits_per_component == 16 )
    {
        pitch        = width*num_components*sizeof(uint16_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else
    {
        throw Exception( "Unsupported bits/component in glTF image" );
    }


    cudaArray_t   cuda_array = nullptr;
    CUDA_CHECK( cudaMallocArray(
                &cuda_array,
                &channel_desc,
                width,
                height
                ) );
    CUDA_CHECK( cudaMemcpy2DToArray(
                cuda_array,
                0,     // X offset
                0,     // Y offset
                data,
                pitch,
                pitch,
                height,
                cudaMemcpyHostToDevice
                ) );
    m_images.push_back( cuda_array );
}


 void MulticamScene::addSampler(
         cudaTextureAddressMode address_s,
         cudaTextureAddressMode address_t,
         cudaTextureFilterMode  filter,
         const int32_t          image_idx
         )
{
    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = getImage( image_idx );

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = address_s == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                   address_s == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                     cudaAddressModeWrap;
    tex_desc.addressMode[1]      = address_t == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                   address_t == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                     cudaAddressModeWrap;
    tex_desc.filterMode          = filter    == GL_NEAREST         ? cudaFilterModePoint   :
                                                                     cudaFilterModeLinear;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK( cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr ) );
    m_samplers.push_back( cuda_tex );
}


CUdeviceptr MulticamScene::getBuffer( int32_t buffer_index ) const
{
    return m_buffers[ buffer_index ];
}


cudaArray_t MulticamScene::getImage( int32_t image_index ) const
{
    return m_images[ image_index ];
}


cudaTextureObject_t MulticamScene::getSampler( int32_t sampler_index ) const
{
    return m_samplers[ sampler_index ];
}


void MulticamScene::finalize()
{
    createContext();
    buildMeshAccels();
    buildInstanceAccel();
    //createPTXModule(m_compound_ptx_module, "ommatidialShader.cu");
    //createPTXModule(m_ptx_module, "shaders.cu");
    createPTXModule();
    createProgramGroups();
    createPipeline();
    createCompoundPipeline();
    // Now handle the creation of the standard SBT table:
    createSBTmissAndHit(m_sbt);
    // Make sure the raygenRecord is pointed at and valid memory:
    GenericCamera* c = getCamera();
    c->forcePackAndCopyRecord(m_raygen_prog_group);
    m_sbt.raygenRecord = c->getRecordPtr();

    // Now handle the creation of the compound SBT:
    createSBTmissAndHit(m_compound_sbt);
    // Make sure the raygenRecord is pointed at and valid memory:
    regenerateCompoundRaygenRecord();

    m_scene_aabb.invalidate();
    for( const auto mesh: m_meshes )
        m_scene_aabb.include( mesh->world_aabb );

    //if( !m_cameras.empty() )
    //    m_cameras.front().setLookat( m_scene_aabb.center() );
}


MulticamScene::~MulticamScene()
{
  cleanup();
}

void MulticamScene::cleanup()
{
    // TODO
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_pinhole_record)));
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_ortho_record)));
    //TODO: destroy the camera vector properly
    freeCompoundBuffer();
}

//------------------------------------------------------------------------------
//
//  CAMERA FUNCTIONS
//
//------------------------------------------------------------------------------

void MulticamScene::addCamera(GenericCamera* cameraPtr)
{
  m_cameras.push_back(cameraPtr);
  checkIfCurrentCameraIsCompound();
}
GenericCamera* MulticamScene::getCamera()
{
  if(!m_cameras.empty())
  {
    return m_cameras[currentCamera];
  }

  std::cerr << "Initializing default camera" << std::endl;
  //cam.setFovY( 45.0f );
  //cam.setLookat( m_scene_aabb.center() );
  //cam.setEye   ( m_scene_aabb.center() + make_float3( 0.0f, 0.0f, 1.5f*m_scene_aabb.maxExtent() ) );

  PerspectiveCamera* cam = new PerspectiveCamera("Default Camera");
  this->addCamera(cam);
  return getCamera();

}
void MulticamScene::setCurrentCamera(const int index)
{
  const int s = int(getCameraCount());
  currentCamera = (index%s + s)%s;
  checkIfCurrentCameraIsCompound();
}
const size_t MulticamScene::getCameraCount() const
{
  return m_cameras.size();
}
void MulticamScene::nextCamera()
{
  setCurrentCamera(currentCamera+1);
}
void MulticamScene::previousCamera()
{
  setCurrentCamera(currentCamera-1);
}

//------------------------------------------------------------------------------
//
//  COMPOUND EYE FUNCTIONS
//
//------------------------------------------------------------------------------
uint32_t MulticamScene::addCompoundCamera(CompoundEye* cameraPtr)
{
  m_compoundEyes.push_back(cameraPtr);
  updateCompoundDataCache();
  return (m_compoundEyes.size()-1);
}
void MulticamScene::checkIfCurrentCameraIsCompound()
{
  GenericCamera* cam = getCamera();
  bool out = false;
  for(size_t i = 0; i<m_compoundEyes.size(); i++)
    out |= cam == m_compoundEyes[i];
  m_selectedCameraIsCompound = out;
}
void MulticamScene::updateCompoundDataCache()
{
  // Update size information
  m_compoundBufferHeight = m_compoundEyes.size();
  uint32_t maxWidth = 0;
  uint32_t maxDepth = 0;
  for(size_t i = 0; i<m_compoundEyes.size(); i++)
  {
    maxWidth = max(maxWidth, m_compoundEyes[i]->getOmmatidialCount());
    maxDepth = max(maxDepth, m_compoundEyes[i]->getSamplesPerOmmatidium());
  }
  m_compoundBufferWidth = maxWidth;
  m_compoundBufferDepth = maxDepth;
  // Update the pointers
  freeCompoundBuffer();
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_compoundBuffer ), sizeof(float3)*m_compoundBufferWidth*m_compoundBufferHeight*m_compoundBufferDepth) );
  freeRandomBuffer();
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_randomStateBuffer ), sizeof(curandState)*m_compoundBufferWidth*m_compoundBufferHeight*m_compoundBufferDepth) );
  // TODO: The randomStateBuffer is currently unitialized. For now we'll be initializing it with if statements in the ommatidial shader, but in the future a CUDA function could be called here to initialize it.
}
void MulticamScene::getCompoundBufferInfo(CUdeviceptr& ptr, uint32_t& width, uint32_t& height, uint32_t& depth, CUdeviceptr& randoPtr) const
{
  ptr = d_compoundBuffer;
  width = m_compoundBufferWidth;
  height = m_compoundBufferHeight;
  depth = m_compoundBufferDepth;
  randoPtr = d_randomStateBuffer;
}
void MulticamScene::freeCompoundBuffer()
{
  if(d_compoundBuffer != 0)
  {
    // Deallocate the buffer if it exists
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_compoundBuffer)) );
  }
}
void MulticamScene::emptyCompoundBuffer()
{
  if(d_compoundBuffer != 0)
  {
    // Copy in zeros if the buffer exists
    CUDA_CHECK( cudaMemset(reinterpret_cast<void*>(d_compoundBuffer), 0, sizeof(float3)*m_compoundBufferWidth*m_compoundBufferHeight*m_compoundBufferDepth) );
    CUDA_SYNC_CHECK();
  }
}
void MulticamScene::freeRandomBuffer()
{
  if(d_randomStateBuffer != 0)
  {
    // Deallocate the buffer if it exists
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_randomStateBuffer)) );
  }
}

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void MulticamScene::createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );

    CUcontext          cuCtx = nullptr;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
}

namespace {
template <typename T = char>
class CuBuffer
{
  public:
    CuBuffer( size_t count = 0 ) { alloc( count ); }
    ~CuBuffer() { free(); }
    void alloc( size_t count )
    {
        free();
        m_allocCount = m_count = count;
        if( m_count )
        {
            CUDA_CHECK( cudaMalloc( &m_ptr, m_allocCount * sizeof( T ) ) );
        }
    }
    void allocIfRequired( size_t count )
    {
        if( count <= m_count )
        {
            m_count = count;
            return;
        }
        alloc( count );
    }
    CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>( m_ptr ); }
    CUdeviceptr get( size_t index ) const { return reinterpret_cast<CUdeviceptr>( m_ptr + index ); }
    void        free()
    {
        m_count      = 0;
        m_allocCount = 0;
        CUDA_CHECK( cudaFree( m_ptr ) );
        m_ptr = nullptr;
    }
    CUdeviceptr release()
    {
        CUdeviceptr current = reinterpret_cast<CUdeviceptr>( m_ptr );
        m_count             = 0;
        m_allocCount        = 0;
        m_ptr               = nullptr;
        return current;
    }
    void upload( const T* data )
    {
        CUDA_CHECK( cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice ) );
    }

    void download( T* data ) const
    {
        CUDA_CHECK( cudaMemcpy( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    void downloadSub( size_t count, size_t offset, T* data ) const
    {
        assert( count + offset < m_allocCount );
        CUDA_CHECK( cudaMemcpy( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }
    size_t count() const { return m_count; }
    size_t reservedCount() const { return m_allocCount; }
    size_t byteSize() const { return m_allocCount * sizeof( T ); }

  private:
    size_t m_count      = 0;
    size_t m_allocCount = 0;
    T*     m_ptr        = nullptr;
};
}  // namespace

void MulticamScene::buildMeshAccels( uint32_t triangle_input_flags )
{
    // Problem:
    // The memory requirements of a compacted GAS are unknown prior to building the GAS.
    // Hence, compaction of a GAS requires to build the GAS first and allocating memory for the compacted GAS afterwards.
    // This causes a device-host synchronization point, potentially harming performance.
    // This is most likely the case for small GASes where the actual building and compaction of the GAS is very fast.
    // A naive algorithm processes one GAS at a time with the following steps:
    // 1. compute memory sizes for the build process (temporary buffer size and build buffer size)
    // 2. allocate temporary and build buffer
    // 3. build the GAS (with temporary and build buffer) and compute the compacted size
    // If compacted size is smaller than build buffer size (i.e., compaction is worth it):
    // 4. allocate compacted buffer (final output buffer)
    // 5. compact GAS from build buffer into compact buffer
    //
    // Idea of the algorithm:
    // Batch process the building and compaction of multiple GASes to avoid host-device synchronization.
    // Ideally, the number of synchronization points would be linear with the number of batches rather than the number of GASes.
    // The main constraints for selecting batches of GASes are:
    // a) the peak memory consumption when batch processing GASes, and
    // b) the amount of memory for the output buffer(s), containing the compacted GASes. This is also part of a), but is also important after the build process.
    // For the latter we try to keep it as minimal as possible, i.e., the total memory requirements for the output should equal the sum of the compacted sizes of the GASes.
    // Hence, it should be avoided to waste memory by allocating buffers that are bigger than what is required for a compacted GAS.
    //
    // The peak memory consumption effectively defines the efficiency of the algorithm.
    // If memory was unlimited, compaction isn't needed at all.
    // A lower bound for the peak memory consumption during the build is the output of the process, the size of the compacted GASes.
    // Peak memory consumption effectively defines the memory pool available during the batch building and compaction of GASes.
    //
    // The algorithm estimates the size of the compacted GASes by a give compaction ratio as well as the computed build size of each GAS.
    // The compaction ratio is defined as: size of compacted GAS / size of build output of GAS.
    // The validity of this estimate therefore depends on the assumed compaction ratio.
    // The current algorithm assumes a fixed compaction ratio.
    // Other strategies could be:
    // - update the compaction ration on the fly by do statistics on the already processed GASes to have a better guess for the remaining batches
    // - multiple compaction rations by type of GAS (e.g., motion vs static), since the type of GAS impacts the compaction ratio
    // Further, compaction may be skipped for GASes that do not benefit from compaction (compaction ratio of 1.0).
    //
    // Before selecting GASes for a batch, all GASes are sorted by size (their build size).
    // Big GASes are handled before smaller GASes as this will increase the likelihood of the peak memory consumption staying close to the minimal memory consumption.
    // This also increase the benefit of batching since small GASes that benefit most from avoiding synchronizations are built "together".
    // The minimum batch size is one GAS to ensure forward process.
    //
    // Goal:
    // Estimate the required output size (the minimal peak memory consumption) and work within these bounds.
    // Batch process GASes as long as they are expected to fit into the memory bounds (non strict).
    //
    // Assumptions:
    // The inputs to each GAS are already in device memory and are needed afterwards.
    // Otherwise this could be factored into the peak memory consumption.
    // E.g., by uploading the input data to the device only just before building the GAS and releasing it right afterwards.
    //
    // Further, the peak memory consumption of the application / system is influenced by many factors unknown to this algorithm.
    // E.g., if it is known that a big pool of memory is needed after GAS building anyways (e.g., textures that need to be present on the device),
    // peak memory consumption will be higher eventually and the GAS build process could already make use of a bigger memory pool.
    //
    // TODOs:
    // - compaction ratio estimation / updating
    // - handling of non-compactable GASes
    // - integration of GAS input data upload / freeing
    // - add optional hard limits / check for hard memory limits (shrink batch size / abort, ...)
    //////////////////////////////////////////////////////////////////////////

    // Magic constants:

    // see explanation above
    constexpr double initialCompactionRatio = 0.5;

    // It is assumed that trace is called later when the GASes are still in memory.
    // We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
    // Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
    constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

    //////////////////////////////////////////////////////////////////////////

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    struct GASInfo {
        std::vector<OptixBuildInput> buildInputs;
        OptixAccelBufferSizes gas_buffer_sizes;
        std::shared_ptr<MeshGroup> mesh;
    };
    std::multimap<size_t, GASInfo> gases;
    size_t totalTempOutputSize = 0;

    for(size_t i=0; i<m_meshes.size(); ++i)
    {
        auto& mesh = m_meshes[i];

        const size_t num_subMeshes =  mesh->indices.size();
        std::vector<OptixBuildInput> buildInputs(num_subMeshes);

        assert(mesh->positions.size() == num_subMeshes &&
            mesh->normals.size()   == num_subMeshes &&
            mesh->texcoords.size() == num_subMeshes);

        for(size_t i = 0; i < num_subMeshes; ++i)
        {
            OptixBuildInput& triangle_input                          = buildInputs[i];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes         =
                mesh->positions[i].byte_stride ?
                mesh->positions[i].byte_stride :
                sizeof(float3),
                triangle_input.triangleArray.numVertices             = mesh->positions[i].count;
            triangle_input.triangleArray.vertexBuffers               = &(mesh->positions[i].data);
            triangle_input.triangleArray.indexFormat                 =
                mesh->indices[i].elmt_byte_size == 2 ?
                OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :
                OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes          =
                mesh->indices[i].byte_stride ?
                mesh->indices[i].byte_stride :
                mesh->indices[i].elmt_byte_size*3;
            triangle_input.triangleArray.numIndexTriplets            = mesh->indices[i].count / 3;
            triangle_input.triangleArray.indexBuffer                 = mesh->indices[i].data;
            triangle_input.triangleArray.flags                       = &triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords               = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &accel_options, buildInputs.data(),
                                                   static_cast<unsigned int>( num_subMeshes ), &gas_buffer_sizes ) );

        totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
        GASInfo g = {std::move( buildInputs ), gas_buffer_sizes, mesh};
        gases.emplace( gas_buffer_sizes.outputSizeInBytes, g );
    }

    size_t totalTempOutputProcessedSize = 0;
    size_t usedCompactedOutputSize = 0;
    double compactionRatio = initialCompactionRatio;

    CuBuffer<char> d_temp;
    CuBuffer<char> d_temp_output;
    CuBuffer<size_t> d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    while( !gases.empty() )
    {
        // The estimated total output size that we end up with when using compaction.
        // It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
        // Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
        size_t remainingEstimatedTotalOutputSize =
            ( size_t )( ( totalTempOutputSize - totalTempOutputProcessedSize ) * compactionRatio );
        size_t availableMemPoolSize = remainingEstimatedTotalOutputSize + additionalAvailableMemory;
        // We need to fit the following things into availableMemPoolSize:
        // - temporary buffer for building a GAS (only during build, can be cleared before compaction)
        // - build output buffer of a GAS
        // - size (actual number) of a compacted GAS as output of a build
        // - compacted GAS

        size_t batchNGASes                    = 0;
        size_t batchBuildOutputRequirement    = 0;
        size_t batchBuildMaxTempRequirement   = 0;
        size_t batchBuildCompactedRequirement = 0;
        for( auto it = gases.rbegin(); it != gases.rend(); it++ )
        {
            batchBuildOutputRequirement += it->second.gas_buffer_sizes.outputSizeInBytes;
            batchBuildCompactedRequirement += ( size_t )( it->second.gas_buffer_sizes.outputSizeInBytes * compactionRatio );
            // roughly account for the storage of the compacted size, although that goes into a separate buffer
            batchBuildOutputRequirement += 8ull;
            // make sure that all further output pointers are 256 byte aligned
            batchBuildOutputRequirement = roundUp<size_t>( batchBuildOutputRequirement, 256ull );
            // temp buffer is shared for all builds in the batch
            batchBuildMaxTempRequirement = std::max( batchBuildMaxTempRequirement, it->second.gas_buffer_sizes.tempSizeInBytes );
            batchNGASes++;
            if( ( batchBuildOutputRequirement + batchBuildMaxTempRequirement + batchBuildCompactedRequirement ) > availableMemPoolSize )
                break;
        }

        // d_temp may still be available from a previous batch, but is freed later if it is "too big"
        d_temp.allocIfRequired( batchBuildMaxTempRequirement );

        // trash existing buffer if it is more than 10% bigger than what we need
        // if it is roughly the same, we keep it
        if( d_temp_output.byteSize() > batchBuildOutputRequirement * 1.1 )
            d_temp_output.free();
        d_temp_output.allocIfRequired( batchBuildOutputRequirement );

        // this buffer is assumed to be very small
        // trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
        if( d_temp_compactedSizes.reservedCount() > batchNGASes * 2 && d_temp_compactedSizes.byteSize() > 20 * 1024 * 1024 )
            d_temp_compactedSizes.free();
        d_temp_compactedSizes.allocIfRequired( batchNGASes );

        // sum of build output size of GASes, excluding alignment
        size_t batchTempOutputSize = 0;
        // sum of size of compacted GASes
        size_t batchCompactedSize = 0;

        auto it = gases.rbegin();
        for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
        {
            emitProperty.result = d_temp_compactedSizes.get( i );
            GASInfo& info = it->second;

            OPTIX_CHECK( optixAccelBuild( m_context, 0,   // CUDA stream
                                            &accel_options,
                                            info.buildInputs.data(),
                                            static_cast<unsigned int>( info.buildInputs.size() ),
                                            d_temp.get(),
                                            d_temp.byteSize(),
                                            d_temp_output.get( tempOutputAlignmentOffset ),
                                            info.gas_buffer_sizes.outputSizeInBytes,
                                            &info.mesh->gas_handle,
                                            &emitProperty,  // emitted property list
                                            1               // num emitted properties
                                            ) );

            tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
            it++;
        }

        // trash d_temp if it is at least 20MB in size
        if( d_temp.byteSize() > 20 * 1024 * 1024 )
            d_temp.free();

        // download all compacted sizes to allocate final output buffers for these GASes
        std::vector<size_t> h_compactedSizes( batchNGASes );
        d_temp_compactedSizes.download( h_compactedSizes.data() );

        //////////////////////////////////////////////////////////////////////////
        // TODO:
        // Now we know the actual memory requirement of the compacted GASes.
        // Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
        bool canCompact = false;
        it = gases.rbegin();
        for( size_t i = 0; i < batchNGASes; ++i )
        {
            GASInfo& info = it->second;
            if( info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i] )
            {
                canCompact = true;
                break;
            }
            it++;
        }

        if( canCompact )
        {
            //////////////////////////////////////////////////////////////////////////
            // "batch allocate" the compacted buffers
            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                batchCompactedSize += h_compactedSizes[i];
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &info.mesh->d_gas_output ), h_compactedSizes[i] ) );
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
                it++;
            }

            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                OPTIX_CHECK( optixAccelCompact( m_context, 0, info.mesh->gas_handle, info.mesh->d_gas_output,
                                                h_compactedSizes[i], &info.mesh->gas_handle ) );
                it++;
            }
        }
        else
        {
            it = gases.rbegin();
            for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                info.mesh->d_gas_output = d_temp_output.get( tempOutputAlignmentOffset );
                batchCompactedSize += h_compactedSizes[i];
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

                tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
                it++;
            }
            d_temp_output.release();
        }

        usedCompactedOutputSize += batchCompactedSize;

        gases.erase( it.base(), gases.end() );
    }
}


///TODO
struct Instance
{
    float transform[12];
};

void MulticamScene::buildInstanceAccel( int rayTypeCount )
{
    const size_t num_instances = m_meshes.size();

    std::vector<OptixInstance> optix_instances( num_instances );

    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_meshes.size(); ++i )
    {
        auto  mesh = m_meshes[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy( optix_instance.transform, mesh->transform.getData(), sizeof( float ) * 12 );

        sbt_offset += static_cast<unsigned int>( mesh->indices.size() ) * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * num_instances;
    CUdeviceptr  d_instances;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_instances ),
                optix_instances.data(),
                instances_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( num_instances );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                m_context,
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer ),
                ias_buffer_sizes.tempSizeInBytes
                ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_d_ias_output_buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                m_context,
                nullptr,                  // CUDA stream
                &accel_options,
                &instance_input,
                1,                  // num build inputs
                d_temp_buffer,
                ias_buffer_sizes.tempSizeInBytes,
                m_d_ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &m_ias_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
}

void MulticamScene::createPTXModule()
{

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur            = false;
    m_pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues          = globalParameters::NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues        = 2; // todo
    m_pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE; // should be optix_exception_flag_stack_overflow;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = getPtxString( "guiEyeRenderer3", "shaders.cu" );

    m_ptx_module  = {};
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                m_context,
                &module_compile_options,
                &m_pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &m_ptx_module
                ) );
}


void MulticamScene::createProgramGroups()
{
     char log[2048];
     size_t sizeof_log = sizeof( log );

     {
         // Create the ommatidial raygen group
         OptixProgramGroupDesc compound_prog_group_desc    = {};
         compound_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
         compound_prog_group_desc.raygen.module            = m_ptx_module;
         compound_prog_group_desc.raygen.entryFunctionName = "__raygen__ommatidium";

         OPTIX_CHECK_LOG( optixProgramGroupCreate(
                     m_context,
                     &compound_prog_group_desc,
                     1,                             // num program groups
                     &program_group_options,
                     log,
                     &sizeof_log,
                     &m_compound_raygen_group
                     )
                 );
     }

     {
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = GenericCamera::DEFAULT_RAYGEN_PROGRAM;

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &raygen_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &m_raygen_prog_group
                    )
                );
    }


    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_radiance";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &m_radiance_miss_group
                    )
                );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &m_occlusion_miss_group
                    )
                );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &hit_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &m_radiance_hit_group
                    )
                );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
                    m_context,
                    &hit_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &m_occlusion_hit_group
                    )
                );
    }
}


void MulticamScene::createPipeline()
{
    OptixProgramGroup program_groups[] =
    {
        m_raygen_prog_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                m_context,
                &m_pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &m_pipeline
                ) );
}

void MulticamScene::createCompoundPipeline()
{
    std::cout<<"raygen: "<<m_raygen_prog_group<<std::endl;
    std::cout<<"compou: "<<m_compound_raygen_group<<std::endl;
    OptixProgramGroup program_groups[] =
    {
        m_compound_raygen_group,
        //m_raygen_prog_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                m_context,
                &m_pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &m_compound_pipeline
                ) );
}

// TODO: Function here that swaps the contents of m_sbt.raygenRecord?
// Perhaps assign some CUdevicePointr to the pinhole sbt record, one for the ortho,
// Then below configures both of those and sets raygenRecord to the first.
// Then we dynamically define the insect eye one in.. some other way.

// I think the cameras themselves should handle the construction and management of the sbt records

void MulticamScene::reconfigureSBTforCurrentCamera()
{
  GenericCamera* c = getCamera();
  char log[2048];
  size_t sizeof_log = sizeof( log );

  // Here, we regenerate the raygen pipeline if the camera has changed types:
  if(getCameraIndex() != lastPipelinedCamera || lastPipelinedCamera == -1)
  {
    lastPipelinedCamera = currentCamera;// update the pointer
    raygen_prog_group_desc.raygen.entryFunctionName = c->getEntryFunctionName();
    std::cout<< "ALERT: Regenerating pipeline with raygen entry function '"<<c->getEntryFunctionName()<<"'."<<std::endl;
    optixProgramGroupDestroy(m_raygen_prog_group);
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                m_context,
                &raygen_prog_group_desc,
                1,                             // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_raygen_prog_group
                )
            );

    c->packAndCopyRecordIfChanged(m_raygen_prog_group);
    m_sbt.raygenRecord = c->getRecordPtr();

    optixPipelineDestroy(m_pipeline);
    createPipeline();
  }else{
    // If the camera's on-device memory has been updated host-side, then re-sync it with the device:
    c->packAndCopyRecordIfChanged(m_raygen_prog_group);
  }
}

void MulticamScene::regenerateCompoundRaygenRecord()
{
  // Assemble the contents of the compound raygen record
  size_t eyeCount = m_compoundEyes.size();
  m_eyeCollectionRecord.data.eyeCount = eyeCount;// Set the number of compound eyes
  //// Construct data.d_list_of_compound_eyes as a list of CUdeviceptrs to CompoundEyeRecords already on-device
  //// (Note that as the on-device records get updated, these pointers will remain valid unless they are reallocated)
  // But first check if the list of compound eyes is allocated
  if(m_eyeCollectionRecord.data.d_compoundEyes == 0)
  {
    std::cout<<"Allocating eye collection on VRAM"<<std::endl;
    // If it isnt', then allocate it on-device:
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&(m_eyeCollectionRecord.data.d_compoundEyes)),
                            sizeof(CUdeviceptr) * eyeCount) );
  }
  /// Then create a list of pointers to each record and copy it over to the device:
  CUdeviceptr eyeData[eyeCount];
  for(size_t i = 0; i<eyeCount; i++)
    eyeData[i] = m_compoundEyes[i]->getRecordPtr();
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>(m_eyeCollectionRecord.data.d_compoundEyes),
              &eyeData[0],
              sizeof(CUdeviceptr)*eyeCount,
              cudaMemcpyHostToDevice
              )
            );

  //// After the list of compound eyes has been copied into VRAM, push the new data to the SBT record (consisting of a device-side pointer to the data and a count of the insect eyes in it)
  // First check if the device-side record exists:
  if(d_eyeCollectionRecord == 0)
  {
    std::cout<<"Allocating eye collection *record* on VRAM"<<std::endl;
    // Make it if it doesn't
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_eyeCollectionRecord), sizeof(EyeCollectionRecord)) );
  }
  std::cout<<"copying eyes *record* to VRAM"<<std::endl;
  // Then pack and copy the record across
  OPTIX_CHECK( optixSbtRecordPackHeader(m_compound_raygen_group, &m_eyeCollectionRecord) ); // Pack the record
  // Copy m_eyeCollectionRecord into d_eyeCollectionRecord:
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_eyeCollectionRecord ),
              &m_eyeCollectionRecord,
              sizeof(EyeCollectionRecord),
              cudaMemcpyHostToDevice
              ) );
  m_compound_sbt.raygenRecord = d_eyeCollectionRecord; // Set the raygen record in m_compound_sbt
}
void MulticamScene::createSBTmissAndHit(OptixShaderBindingTable& sbt)
{
    // Per-camera raygen Records are handled by each camera

    // Miss Record
    {
        const size_t miss_record_size = sizeof( EmptyRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &sbt.missRecordBase ),
                    miss_record_size*globalParameters::RAY_TYPE_COUNT
                    ) );

        EmptyRecord ms_sbt[ globalParameters::RAY_TYPE_COUNT ];
        OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_miss_group,  &ms_sbt[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_miss_group, &ms_sbt[1] ) );

        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( sbt.missRecordBase ),
                    ms_sbt,
                    miss_record_size*globalParameters::RAY_TYPE_COUNT,
                    cudaMemcpyHostToDevice
                    ) );
        sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        sbt.missRecordCount     = globalParameters::RAY_TYPE_COUNT;
    }

    // Hitgroup Records
    {
        std::vector<HitGroupRecord> hitgroup_records;
        for( const auto mesh : m_meshes )
        {
            for( size_t i = 0; i < mesh->material_idx.size(); ++i )
            {
                HitGroupRecord rec = {};
                OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_hit_group, &rec ) );
                rec.data.geometry_data.type                    = GeometryData::TRIANGLE_MESH;
                rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
                rec.data.geometry_data.triangle_mesh.normals   = mesh->normals[i];
                rec.data.geometry_data.triangle_mesh.texcoords = mesh->texcoords[i];
                rec.data.geometry_data.triangle_mesh.indices   = mesh->indices[i];

                const int32_t mat_idx  = mesh->material_idx[i];
                if( mat_idx >= 0 )
                    rec.data.material_data.pbr = m_materials[ mat_idx ];
                else
                    rec.data.material_data.pbr = MaterialData::Pbr();
                hitgroup_records.push_back( rec );

                OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_hit_group, &rec ) );
                hitgroup_records.push_back( rec );
            }
        }

        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &sbt.hitgroupRecordBase ),
                    hitgroup_record_size*hitgroup_records.size()
                    ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                    hitgroup_records.data(),
                    hitgroup_record_size*hitgroup_records.size(),
                    cudaMemcpyHostToDevice
                    ) );

        sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( hitgroup_record_size );
        sbt.hitgroupRecordCount         = static_cast<unsigned int>( hitgroup_records.size() );
    }
}
