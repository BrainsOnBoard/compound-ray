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

#define DEBUG

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "eyeRenderer.h"
#include "BillboardPrimitive.h"
#include "TriangleMeshObject.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

// SBT Definitions
#include "SbtRecord.h"
#include "TestObjectSbt.h"

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 1.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 0.0f} );
    cam.setFovY( 120);//45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width  = 1024;
    int         height =  768;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
          // Initialize CUDA
          CUDA_CHECK( cudaFree( 0 ) );

          CUcontext cuCtx = 0;  // zero means take the current context
          OPTIX_CHECK( optixInit() );
          OptixDeviceContextOptions options = {};
          options.logCallbackFunction       = &context_log_cb;
          options.logCallbackLevel          = 4;
          OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }


        // Create a new triangle object on the stack
        TriangleMeshObject box = TriangleMeshObject();

        //
        // accel handling
        //
        //createAccelerationStructure();
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            // Set flags
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // Build mesh data and then assign to device memory
            box.setMeshDataToDefault();
            CUdeviceptr verts = box.copyVerticesToDevice();

            std::cout<<"Box has " << box.getVertexCount() << " vertices." << std::endl;

            // Create a triangle OptixBuildInput object based on the verticies
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = box.getVertexCountUint();
            //triangle_input.triangleArray.indexBuffer   =  // Points at on-device buffer of vertex index forming triangles
            //triangle_input.triangleArray.numIndexTriplets = // The size of the above.
            triangle_input.triangleArray.vertexBuffers = box.getDeviceVertexPointerPointer();
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            // Check how much memory the object will take up on-device
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &triangle_input,
                                                       1,  // Number of build inputs
                                                       &gas_buffer_sizes ) );
            // Allocate assembly size temporary buffer
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

            // Allocate the actual GAS structure, but be ready for it to be smaller than it should
            // non-compacted output
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(
                            &d_buffer_temp_output_gas_and_compacted_size ),
                        compactedSizeOffset + 8 // Add a little space for (*I think*) the emitted response from the device
                        ) );

            // ...Allocate some memory on the device to store the emitted feedback information
            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

            // Build the accelleration structure and ref. to &gas_handle, linkig in everything
            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,              // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,              // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_buffer_temp_output_gas_and_compacted_size,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        &emitProperty,  // emitted property list
                        1               // num emitted properties
                        ) );

            // Free the temporary buffer after it's been used to assemble the GAS (and also the verticies, as they're in the GAS now)
            CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
            box.deleteDeviceVertices(); // These vertices are freed now.

            // Take the feedback information that was emitted, extract the potential compacted size
            size_t compacted_gas_size;
            CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

            // Check if compaction would make the GAS smaller on-device...
            if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
            {
                // If it would, then allocate space for the smaller one...
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

                // ...and compact the GAS into the newly allocated smaller space
                // use handle as input and output
                OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

                // Finally deallocate the temporary size
                CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
            }
            else
            {
                // If compaction doesn't get us any benefit, then just set the output_buffer to point to the same place as the temporary one.
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // Create modules
        // Actually links up the OptiX programs, configures the ray payload data, globally accessible data
        //
        
        // Store pipeline compilation options
        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues      = 3;
        pipeline_compile_options.numAttributeValues    = 3;//max(bbp.getNumberOfRequiredAttributeValues(), 3); // Make sure to get the maximum number of atts.
        pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        OptixModule module = nullptr;
        {
            // Set options
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;


            // Load the PTX string
            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "eyeRenderer.cu" );
            size_t sizeof_log = sizeof( log );

            // Compile the module
            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        ptx.c_str(),
                        ptx.size(),
                        log,
                        &sizeof_log,
                        &module
                        ) );
        }

        // Generate an optix module for the billboard primitive
        //OptixModule billboardModule = bbp.createOptixModule(pipeline_compile_options, &context, log, sizeof(log));


        //
        // Create program groups
        // These are groups of programs that render the GAS
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        OptixProgramGroup intersection_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );

            //OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            //hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            //hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            //hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            //sizeof_log = sizeof( log );
            //OPTIX_CHECK_LOG( optixProgramGroupCreate(
            //            context,
            //            &hitgroup_prog_group_desc,
            //            1,   // num program groups
            //            &program_group_options,
            //            log,
            //            &sizeof_log,
            //            &hitgroup_prog_group
            //            ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            //hitgroup_prog_group_desc.hitgroup.moduleCH            = billboardModule; // In a better one, this would be it's own module made above
            //hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__closehit";

            //bbp.appendIntersection(&hitgroup_prog_group_desc, &billboardModule);

            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &hitgroup_prog_group
                        ) );
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = 5;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            pipeline_link_options.overrideUsesMotionBlur = false;
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &pipeline
                        ) );
        }

        //
        // Set up shader binding table
        // This binds variables to each shader.
        //
        // The binding table is composed of different records:
        // Only one for the raygen, then N each for the miss and hitgroup shaders.
        // It stores the first one of each list in a variable, then the length and count of them
        // Each thing is a device pointer of type CUdeviceptr
        //
        OptixShaderBindingTable sbt = {};
        {
            // Allocate memory on-device for the raygen data type
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            // Configure the data to store in there
            sutil::Camera cam;
            configureCamera( cam, width, height );
            RayGenSbtRecord rg_sbt;
            rg_sbt.data ={};
            rg_sbt.data.cam_eye = cam.eye();
            // Actually put the camera data into the host-side RayGenSbtRecord datatype
            cam.UVWFrame( rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w );
            // Pack the device memory
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            // Copy the data from host to device
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            //CUdeviceptr hitgroup_record;
            //size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
            //CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            //HitGroupSbtRecord hg_sbt;
            //hg_sbt.data = { 1.5f };
            //OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            //CUDA_CHECK( cudaMemcpy(
            //            reinterpret_cast<void*>( hitgroup_record ),
            //            &hg_sbt,
            //            hitgroup_record_size,
            //            cudaMemcpyHostToDevice
            //            ) );

            // Set up the pointer
            CUdeviceptr d_testObject_record;
            size_t      testObject_record_size = sizeof(TestObjectSbtRecord);
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_testObject_record), testObject_record_size));
            // Configure host memory
            TestObjectSbtRecord to_sbt;
            to_sbt.data.r = 0.0f; // Note: could have done to_sbt.data = {0.0f, 1.0f, 0.0f}
            to_sbt.data.g = 1.0f;
            to_sbt.data.b = 1.0f;
            // Pack the header
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &to_sbt));
            // Aaaand copy to the device
            CUDA_CHECK(cudaMemcpy(
                       reinterpret_cast<void*>(d_testObject_record),
                       &to_sbt,
                       testObject_record_size,
                       cudaMemcpyHostToDevice
                       ));

            // Set the binding table components
            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d_testObject_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1 ;
        }

        // Create an shared buffer for the output
        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        // Actually runs the raycasting
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            // These params are globally accessible
            Params params;
            params.image        = output_buffer.map();
            params.image_width  = width;
            params.image_height = height;
            params.origin_x     = width / 2;
            params.origin_y     = height / 2;
            params.handle       = gas_handle; // Passes the handle from before

            // Copy in the global params
            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            // Launch it
            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            // Stop mapping the buffer once the data's been written in
            output_buffer.unmap();
        }

        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::displayBufferFile( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );
            //OPTIX_CHECK( optixModuleDestroy(billboardModule) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
