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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
//#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "MulticamScene.h"
#include "GlobalParameters.h"
#include "BasicController.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty  = false;
bool              drawUI = true;
int32_t           samplesPerOmmatidium = 1;// Samples per pixel are stored in each camera's settings
bool              saveFlag = false;
const int         SAMPLES_PER_PAGE = 1;// The number of samples up or down the image is changed by when pressing the page button

// Camera state
BasicController basicController;
MulticamScene scene;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;
std::chrono::duration<double> totalRenderTime( 0.0 );

globalParameters::LaunchParams*  d_params = nullptr;
globalParameters::LaunchParams   params   = {};
int32_t                 width    = 768;
int32_t                 height   = 768;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        //trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        //trackball.setViewMode( Trackball::LookAtFixed );
        //trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        //trackball.setViewMode( Trackball::EyeFixed );
        //trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    width   = res_x;
    height  = res_y;
    resize_dirty   = true;
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q )
        {
          glfwSetWindowShouldClose( window, true );
        }else if(key == GLFW_KEY_N){
          scene.nextCamera();
        }else if(key == GLFW_KEY_B){
          scene.previousCamera();
        }else if(key == GLFW_KEY_PAGE_UP){
          if(scene.isCompoundEyeActive())
          {
            ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(SAMPLES_PER_PAGE);
            scene.updateCompoundDataCache();
            params.initializeRandos = true;
          }
        }else if(key == GLFW_KEY_PAGE_DOWN){
          if(scene.isCompoundEyeActive())
          {
            ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(-SAMPLES_PER_PAGE);
            scene.updateCompoundDataCache();
            params.initializeRandos = true;
          }
        }else if(key == GLFW_KEY_C){
          saveFlag = true;
        }

        params.frame = 0;
        totalRenderTime = std::chrono::duration<double>(0.0);
    }
    else if( key == GLFW_KEY_G )
    {
        drawUI = !drawUI;
    }
    basicController.ingestKeyAction(key, action);
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    //if(trackball.wheelEvent((int)yscroll))
    //    camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr <<  "Usage  : " << argv0 << " [options]\n";
    std::cerr <<  "Options: --file | -f <filename>      File for image output\n";
    std::cerr <<  "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr <<  "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr <<  "         --model <model.gltf>        Specify model to render (required)\n";
    std::cerr <<  "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( const MulticamScene& scene ) {
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    params.frame = 0;
    params.initializeRandos = true;
    params.lighting = false;

    const float loffset = scene.aabb().maxExtent();

    // TODO: add light support to sutil::Scene
    std::vector<Light::Point> lights(4);
    lights[0].color     = { 1.0f, 1.0f, 0.8f };
    lights[0].intensity = 5.0f;
    lights[0].position  = scene.aabb().center() + make_float3( loffset );
    lights[0].falloff   = Light::Falloff::QUADRATIC;
    lights[1].color     = { 0.8f, 0.8f, 1.0f };
    lights[1].intensity = 3.0f;
    lights[1].position  = scene.aabb().center() + make_float3( -loffset, 0.5f*loffset, -0.5f*loffset  );
    lights[1].falloff   = Light::Falloff::QUADRATIC;
    lights[2].color     = { 1.0f, 1.0f, 0.8f };
    lights[2].intensity = 5.0f;
    lights[2].position  = scene.aabb().center() + make_float3( 0.0f, 4.0f, -5.0f);
    lights[2].falloff   = Light::Falloff::QUADRATIC;
    lights[3].color     = { 1.0f, 1.0f, 0.8f };
    lights[3].intensity = 0.5f;
    lights[3].position  = scene.aabb().center() + make_float3( 1.0f, -6.0f, 0.0f);
    lights[3].falloff   = Light::Falloff::QUADRATIC;

    params.lights.count  = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.lights.data ),
                lights.size() * sizeof( Light::Point )
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( params.lights.data ),
                lights.data(),
                lights.size() * sizeof( Light::Point ),
                cudaMemcpyHostToDevice
                ) );

    params.miss_color   = make_float3( 0.1f );

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( globalParameters::LaunchParams ) ) );

    params.handle = scene.traversableHandle();
}


// Updates the params to acurately reflect the currently selected camera
void handleCameraUpdate( globalParameters::LaunchParams& params )
{
    GenericCamera* camera  = scene.getCamera();

    // Make sure the SBT of the scene is updated for the newly selected camera before launch,
    // also push any changed host-side camera SBT data over to the device.
    scene.reconfigureSBTforCurrentCamera();
    //camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, globalParameters::LaunchParams& params )
{
    handleCameraUpdate( params );
    handleResize( output_buffer );
}


void launchFrame( sutil::CUDAOutputBuffer<uchar4>& output_buffer, const MulticamScene& scene )
{
    // Map and configure memory
    scene.getCompoundBufferInfo(params.compoundBufferPtr, params.compoundBufferWidth, params.compoundBufferHeight, params.compoundBufferDepth, params.randomsBufferPtr);

    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer        = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
                &params,
                sizeof( globalParameters::LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    if(scene.hasCompoundEyes() && scene.isCompoundEyeActive())
    {
      // Launch ommatidial render; renders all compound eyes simultaneously
      OPTIX_CHECK( optixLaunch(
                  scene.compoundPipeline(),
                  0,             // stream
                  reinterpret_cast<CUdeviceptr>( d_params ),
                  sizeof( globalParameters::LaunchParams ),
                  scene.compoundSbt(),
                  params.compoundBufferWidth, // launch width
                  params.compoundBufferHeight, // launch height
                  params.compoundBufferDepth // launch depth
                  ) );
      CUDA_SYNC_CHECK();
      params.frame++;// Increase the frame number
      params.initializeRandos = false;// Make sure that random stream initialization is only ever done once
    }

    // Launch render
    OPTIX_CHECK( optixLaunch(
                scene.pipeline(),
                0,             // stream
                reinterpret_cast<CUdeviceptr>( d_params ),
                sizeof( globalParameters::LaunchParams ),
                scene.sbt(),
                width,  // launch width
                height, // launch height
                1//scene.getCamera()->samplesPerPixel // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


void initCameraState( MulticamScene& scene )
{
    basicController.speed = 0.1f;//1.0f;
    //camera_changed = true;

    //trackball.setCamera( &(scene.getCamera()));
    //trackball.setMoveSpeed( 10.0f );
    //trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    //trackball.setGimbalLock(true);
}


void cleanup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params               ) ) );
    scene.cleanup();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;
    //std::string infile = sutil::sampleDataFilePath( "WaterBottle/WaterBottle.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "cube/cube.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "roth/glTF/roth.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "Duck/Duck.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "suzanne/suzanne.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "roth/flight-1/flight-1.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "test-scene/test-scene.gltf" );
    std::string infile = sutil::sampleDataFilePath( "ofstad-arena/ofstad-arena.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "roth/flight-2/roth.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "test-scene/disco-cow.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "test-scene/test-scene-no-insect-cam.gltf" );
    //std::string infile = sutil::sampleDataFilePath( "test-scene/test-scene-no-cams.gltf" );

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--model" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            infile = sutil::sampleDataFilePath(argv[++i]);
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
	          std::cout << "Output file changed from \"" << outfile << "\" to \"";
            outfile = argv[++i];
	          std::cout << outfile << "\"\n";
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( infile.empty() )
    {
        std::cerr << "--model argument required" << std::endl;
        printUsageAndExit( argv[0] );
    }


    try
    {
        loadScene( infile.c_str(), scene );
        scene.finalize();

        OPTIX_CHECK( optixInit() ); // Need to initialize function table
        initCameraState( scene );
        initLaunchParams( scene );


        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "Eye Renderer 3.0", width, height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback  ( window, cursorPosCallback   );
            glfwSetWindowSizeCallback ( window, windowSizeCallback  );
            glfwSetKeyCallback        ( window, keyCallback         );
            glfwSetScrollCallback     ( window, scrollCallback      );
            glfwSetWindowUserPointer  ( window, &params       );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );// Output buffer for display
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );


                char cameraInfo[100];
                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();
                    scene.getCamera()->moveLocally(basicController.getMovementVector());
                    scene.getCamera()->rotateLocallyAround(basicController.getVerticalRotationAngle(),  make_float3(1.0f, 0.0f, 0.0f) );
                    scene.getCamera()->rotateAround(basicController.getHorizontalRotationAngle(),make_float3(0.0f, 1.0f, 0.0f) );

                    updateState( output_buffer, params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    // Or maybe instead the filename should be drawn from the cameras themselves, with the compound cameras drawing from a separate file that only contains them?
                    launchFrame( output_buffer, scene );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    totalRenderTime += render_time;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    if(saveFlag)
                    {
                      sutil::ImageBuffer buffer;
                      buffer.data = output_buffer.getHostPointer();
                      buffer.width = output_buffer.width();
                      buffer.height = output_buffer.height();
                      buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
                      sutil::displayBufferFile("output_image.ppm", buffer, false);
                      saveFlag = false;
                    }

                    if(drawUI)
                    {
                      sutil::displayStats( state_update_time, render_time, display_time );

                      double avg = std::chrono::duration_cast<std::chrono::milliseconds>(totalRenderTime).count()/((float)params.frame);
                      sprintf(cameraInfo, "Camera: %i (%s)\nAvg. rendertime: %.1fms", scene.getCameraIndex(), scene.getCamera()->getCameraName(), avg);

                      sutil::beginFrameImGui();
                      sutil::displayText(cameraInfo, 10.0f, 80.0f, 250, 10);
                      sutil::endFrameImGui();
                    }

                    glfwSwapBuffers(window);
                }
                while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
              sutil::initGLFW(); // For GL context
              sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer_single(output_buffer_type, width, height);
            handleCameraUpdate( params);
            handleResize( output_buffer_single );
            launchFrame( output_buffer_single, scene );

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer_single.getHostPointer();
            buffer.width = output_buffer_single.width();
            buffer.height = output_buffer_single.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::displayBufferFile(outfile.c_str(), buffer, false);

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanup();

    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
