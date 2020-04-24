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
#include <optixDemandTexture.h>

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
#include <lib/DemandLoading/EXRReader.h>
#else
#include <lib/DemandLoading/CheckerBoardReader.h>
#endif

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Camera.h>
#include <sutil/sutil.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int32_t width  = 768;
int32_t height = 768;

sutil::Camera camera;

float g_mipLevelBias = 0.0f;

struct PerDeviceSampleState
{
    int32_t                     device_idx               = -1;
    OptixDeviceContext          context                  = 0;
    OptixTraversableHandle      gas_handle               = 0;  // Traversable handle for triangle AS
    CUdeviceptr                 d_gas_output_buffer      = 0;  // Triangle AS memory
    OptixModule                 ptx_module               = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline               pipeline                 = 0;
    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hitgroup_prog_group      = 0;
    OptixShaderBindingTable     sbt                      = {};
    Params                      params;
    Params*                     d_params;
    CUstream                    stream = 0;
};


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "\nUsage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions\n";
    std::cerr << "         --bias | -b                 Mip level bias (default 0.0)\n\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


void initCameraState()
{
    float3 cam_eye = {-3.0f, 0.0f, 0.0f};
    camera.setEye( cam_eye );
    camera.setLookat( make_float3( 0.0f, 0.0f, 0.0f ) );
    camera.setUp( make_float3( 0.0f, 0.0f, 1.0f ) );
    camera.setFovY( 60.0f );
    camera.setAspectRatio( (float)width / (float)height );
}


void getDevices( std::vector<unsigned int>& devices )
{
    int32_t device_count = 0;
    CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
    devices.resize( device_count );
    std::cout << "Total GPUs visible: " << devices.size() << std::endl;
    for( int32_t deviceIndex = 0; deviceIndex < device_count; ++deviceIndex )
    {
        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties ( &prop, deviceIndex ) );
        std::cout << "\t[" << devices[deviceIndex] << "]: " << prop.name << std::endl;
        devices[deviceIndex] = deviceIndex;
    }
}


void createContext( PerDeviceSampleState& state )
{
    // Initialize CUDA on this device
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext        context;
    CUcontext                 cuCtx   = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
}


void createContexts( std::vector<unsigned int>& devices, std::vector<PerDeviceSampleState>& states )
{
    OPTIX_CHECK( optixInit() );

    states.resize( devices.size() );

    for( unsigned int i = 0; i < devices.size(); ++i )
    {
        states[i].device_idx = devices[i];
        CUDA_CHECK( cudaSetDevice( i ) );
        createContext( states[i] );
    }
}


void buildAccel( PerDeviceSampleState& state )
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};

    aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.aabbArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.aabbArray.numPrimitives = 1;

    uint32_t aabb_input_flags[1]       = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.aabbArray.flags         = aabb_input_flags;
    aabb_input.aabbArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,              // CUDA stream
                &accel_options,
                &aabb_input,
                1,              // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,  // emitted property list
                1               // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void createModule( PerDeviceSampleState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount          = 100;
    module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 3;
    state.pipeline_compile_options.numAttributeValues    = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixDemandTexture.cu" );
    char              log[2048];
    size_t            sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX( state.context, &module_compile_options, &state.pipeline_compile_options,
                                               ptx.c_str(), ptx.size(), log, &sizeof_log, &state.ptx_module ) );
}


void createProgramGroups( PerDeviceSampleState& state )
{
    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log                                  = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = state.ptx_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
    sizeof_log                                            = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_group ) );
}


void createPipeline( PerDeviceSampleState& state )
{
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 5;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur   = false;
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ), log,
                                          &sizeof_log, &state.pipeline ) );
}


void createSBT( PerDeviceSampleState& state, const DemandTexture& texture, float texture_scale, float texture_lod )
{
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

    // The demand-loaded texture id is passed to the closest hit program via the hitgroup record.
    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data = {1.5f /*radius*/, texture.getId(), texture_scale, texture_lod};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hg_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice ) );

    state.sbt.raygenRecord                = raygen_record;
    state.sbt.missRecordBase              = miss_record;
    state.sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    state.sbt.hitgroupRecordCount         = 1;
}

void cleanupState( PerDeviceSampleState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}


void initLaunchParams( PerDeviceSampleState& state, DemandTextureManager& textureManager, unsigned int numDevices )
{
    state.params.image_width  = width;
    state.params.image_height = height;
    state.params.origin_x     = width / 2;
    state.params.origin_y     = height / 2;
    state.params.handle       = state.gas_handle;
    state.params.device_idx   = state.device_idx;
    state.params.num_devices  = numDevices;
    state.params.mipLevelBias = g_mipLevelBias;

    state.params.eye = camera.eye();
    camera.UVWFrame( state.params.U, state.params.V, state.params.W );

    state.params.pagingContext = textureManager.getPagingContext( state.device_idx );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
}


void performLaunches( sutil::CUDAOutputBuffer<uchar4>& output_buffer, std::vector<PerDeviceSampleState>& states, DemandTextureManager& textureManager )
{
    for( auto& state : states )
    {
        CUDA_CHECK( cudaSetDevice( state.device_idx ) );

        textureManager.launchPrepare( state.device_idx );
        if( state.params.demandTextures != textureManager.getSamplers( state.device_idx ) )
        {
            state.params.demandTextures = textureManager.getSamplers( state.device_idx );
            CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ), &state.params,
                                         sizeof( state.params ), cudaMemcpyHostToDevice, state.stream ) );
        }

        uchar4* result_buffer_data = output_buffer.map();

        state.params.result_buffer = result_buffer_data;
        CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ), &state.params, sizeof( Params ),
                                     cudaMemcpyHostToDevice, state.stream ) );

        OPTIX_CHECK( optixLaunch( state.pipeline, state.stream, reinterpret_cast<CUdeviceptr>( state.d_params ),
                                  sizeof( Params ), &state.sbt,
                                  state.params.image_height / static_cast<unsigned int>( states.size() ),  // launch height (launch is split across GPUs)
                                  state.params.image_width,                   // launch width
                                  1                                           // launch depth
                                  ) );

        output_buffer.unmap();
    }
    for( auto& state : states )
    {
        CUDA_CHECK( cudaSetDevice( state.device_idx ) );
        CUDA_SYNC_CHECK();
    }
}


int main( int argc, char* argv[] )
{
    std::string outfile;
    std::string textureFile = "Textures/Bricks12_col.exr";

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
                outfile = argv[++i];
            else
                printUsageAndExit( argv[0] );
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else if( arg == "--bias" || arg == "-b" )
        {
            if( i < argc - 1 )
                g_mipLevelBias = static_cast<float>( atof( argv[++i] ) );
            else
                printUsageAndExit( argv[0] );
        }
        else if ( arg == "--texture" || arg == "-t" )
        {
            if( i < argc - 1 )
                textureFile = argv[++i];
            else
                printUsageAndExit( argv[0] );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        std::vector<unsigned int> available_devices;
        getDevices( available_devices );

        std::vector<PerDeviceSampleState> states;
        createContexts( available_devices, states );

        //
        // Initialize DemandTextureManager and create a demand-loaded texture.
        // The texture id is passed to the closest hit shader via a hit group record in the SBT.
        // The texture sampler array (indexed by texture id) is passed as a launch parameter.
        //
        DemandTextureManager textureManager( available_devices );

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
        // Image credit: CC0Textures.com (https://cc0textures.com/view.php?tex=Bricks12)
        // Licensed under the Creative Commons CC0 License.
        // TODO: Should we aggregate all of the sample textures to one folder?
        std::string textureFilename( sutil::sampleDataFilePath( textureFile.c_str() ) );

        std::shared_ptr<EXRReader> textureReader( std::make_shared<EXRReader>( textureFilename.c_str() ) );
        const float                texture_scale = 4.f;
#else
        // If OpenEXR is not available, use a procedurally generated image.
        std::shared_ptr<CheckerBoardReader> textureReader( std::make_shared<CheckerBoardReader>( 1024, 1024 ) );
        const float                      texture_scale = 1.f;
#endif

        const DemandTexture& texture = textureManager.createTexture( textureReader );

        //
        // Set up OptiX state
        //
        for( auto& state : states )
        {
            CUDA_CHECK( cudaSetDevice( state.device_idx ) );
            buildAccel( state );
            createModule( state );
            createProgramGroups( state );
            createPipeline( state );
            createSBT( state, texture, texture_scale, 0.f /*texture_lod*/ );
        }

        for( auto& state : states )
            initLaunchParams( state, textureManager, static_cast<unsigned int>( states.size() ) );

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::ZERO_COPY, width, height );

        performLaunches( output_buffer, states, textureManager );

        for( int numFilled = textureManager.processRequests(); numFilled > 0; numFilled = textureManager.processRequests() )
        {
            std::cout << "Filled " << numFilled << " requests.  Relaunching..." << std::endl;
            performLaunches( output_buffer, states, textureManager );
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

        for( auto& state : states )
            cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
