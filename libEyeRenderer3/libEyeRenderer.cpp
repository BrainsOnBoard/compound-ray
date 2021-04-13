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

#include "libEyeRenderer.h"
//#include <stdio.h>

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

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

MulticamScene scene;

globalParameters::LaunchParams*  d_params = nullptr;
globalParameters::LaunchParams   params   = {};
int32_t                 width    = 400;
int32_t                 height   = 400;

GLFWwindow* window = sutil::initUI( "Eye Renderer 3.0", width, height );
sutil::CUDAOutputBuffer<uchar4> outputBuffer(sutil::CUDAOutputBufferType::GL_INTEROP, width, height);
sutil::GLDisplay gl_display; // Stores the frame buffer to swap in and out

bool notificationsActive = true;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------


//static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
//{
//    width   = res_x;
//    height  = res_y;
//    resize_dirty   = true;
//}


//static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
//{
//    if( action == GLFW_PRESS )
//    {
//        if( key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q )
//        {
//          glfwSetWindowShouldClose( window, true );
//        }else if(key == GLFW_KEY_N){
//          scene.nextCamera();
//        }else if(key == GLFW_KEY_B){
//          scene.previousCamera();
//        }else if(key == GLFW_KEY_PAGE_UP){
//          if(scene.isCompoundEyeActive())
//          {
//            ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(SAMPLES_PER_PAGE);
//            scene.updateCompoundDataCache();
//            params.initializeRandos = true;
//          }
//        }else if(key == GLFW_KEY_PAGE_DOWN){
//          if(scene.isCompoundEyeActive())
//          {
//            ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(-SAMPLES_PER_PAGE);
//            scene.updateCompoundDataCache();
//            params.initializeRandos = true;
//          }
//        }else if(key == GLFW_KEY_C){
//          saveFlag = true;
//        }
//
//        params.frame = 0;
//        totalRenderTime = std::chrono::duration<double>(0.0);
//    }
//    else if( key == GLFW_KEY_G )
//    {
//        drawUI = !drawUI;
//    }
//    basicController.ingestKeyAction(key, action);
//}

void initLaunchParams( const MulticamScene& scene ) {
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    params.frame = 0;
    params.initializeRandos = true;
    params.lighting = false;

    const float loffset = scene.aabb().maxExtent();

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

void cleanup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params               ) ) );
    scene.cleanup();
}

//------------------------------------------------------------------------------
//
// API functions
//
//------------------------------------------------------------------------------
// General Running
//------------------------------------------------------------------------------
void setVerbosity(bool v)
{
  notificationsActive = v;
}
void loadGlTFscene(const char* filepath)
{
  loadScene(filepath, scene);
  scene.finalize();
  initLaunchParams(scene);
}
void setRenderSize(int w, int h)
{
  width = w;
  height = h;
  if(notificationsActive)
    std::cout<<"[PyEye] Resizing rendering buffer to ("<<w<<", "<<h<<")."<<std::endl;
  outputBuffer.resize(width, height);
}
double renderFrame(void)
{
  handleCameraUpdate(params);// Update the params to accurately reflect the currently selected camera

  auto then = std::chrono::steady_clock::now();
  launchFrame( outputBuffer, scene );
  std::chrono::duration<double, std::milli> render_time = std::chrono::steady_clock::now() - then;

  if(notificationsActive)
    std::cout<<"[PyEye] Rendered frame in "<<render_time.count()<<"ms."<<std::endl;

  CUDA_SYNC_CHECK();
  return(render_time.count());
}
void displayFrame(void)
{
  int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
  int framebuf_res_y = 0;   //
  glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
  gl_display.display(
          outputBuffer.width(),
          outputBuffer.height(),
          framebuf_res_x,
          framebuf_res_y,
          outputBuffer.getPBO()
          );

  // Swap the buffer
  glfwSwapBuffers(window);
}
void saveFrameAs(char* ppmFilename)
{
  sutil::ImageBuffer buffer;
  buffer.data = outputBuffer.getHostPointer();
  buffer.width = outputBuffer.width();
  buffer.height = outputBuffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
  sutil::displayBufferFile(ppmFilename, buffer, false);
  if(notificationsActive)
    std::cout<<"[PyEye] Saved render as '"<<ppmFilename<<"'"<<std::endl;
}
unsigned char* getFramePointer(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame pointer..."<<std::endl;
  return (unsigned char*)outputBuffer.getHostPointer();
}
void getFrame(unsigned char* frame)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame..."<<std::endl;
  size_t displaySize = outputBuffer.width()*outputBuffer.height();
  for(size_t i = 0; i<displaySize; i++)
  {
    unsigned char val = (unsigned char)(((float)i/(float)displaySize)*254);
    frame[displaySize*3 + 0] = val;
    frame[displaySize*3 + 1] = val;
    frame[displaySize*3 + 2] = val;
  }
}
void stop(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Cleaning eye renderer resources."<<std::endl;
  sutil::cleanupUI(window);
  cleanup();
}

// C-level only
void * getWindowPointer()
{
  return (void*)window;
}

//------------------------------------------------------------------------------
// Camera Control
//------------------------------------------------------------------------------
size_t getCameraCount()
{
  return(scene.getCameraCount());
}
void nextCamera(void)
{
  scene.nextCamera();
}
size_t getCurrentCameraIndex(void)
{
  return(scene.getCameraIndex());
}
const char* getCurrentCameraName(void)
{
  return(scene.getCamera()->getCameraName());
}
void previousCamera(void)
{
  scene.previousCamera();
}
void gotoCamera(int index)
{
  scene.setCurrentCamera(index);
}
bool gotoCameraByName(char* name)
{
  scene.setCurrentCamera(0);
  for(auto i = 0; i<scene.getCameraCount(); i++)
  {
    if(strcmp(name, scene.getCamera()->getCameraName()) == 0)
      return true;
    scene.nextCamera();
  }
  return false;
}
void setCameraPosition(float x, float y, float z)
{
  scene.getCamera()->setPosition(make_float3(x,y,z));
}
void getCameraPosition(float& x, float& y, float& z)
{
  const float3& camPos = scene.getCamera()->getPosition();
  x = camPos.x;
  y = camPos.y;
  z = camPos.z;
}
void rotateCameraAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateAround(angle,  make_float3(x,y,z));
}
void rotateCameraLocallyAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateLocallyAround(angle,  make_float3(x,y,z));
}
void translateCamera(float x, float y, float z)
{
  scene.getCamera()->move(make_float3(x, y, z));
}
void translateCameraLocally(float x, float y, float z)
{
  scene.getCamera()->moveLocally(make_float3(x, y, z));
}

//------------------------------------------------------------------------------
// Ommatidial Camera Control
//------------------------------------------------------------------------------
bool isCompoundEyeActive(void)
{
  return scene.isCompoundEyeActive();
}
void setCurrentEyeSamplesPerOmmatidium(int s)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->setSamplesPerOmmatidium(s);
    scene.updateCompoundDataCache();
    params.initializeRandos = true;
  }
}
int getCurrentEyeSamplesPerOmmatidium(void)
{
  return(((CompoundEye*)scene.getCamera())->getSamplesPerOmmatidium());
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

//int main(void)
//{
//    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
//
//    //
//    // Parse command line options
//    //
//    std::string outfile;
//    std::string infile = sutil::sampleDataFilePath( "ofstad-arena/ofstad-arena.gltf" );
//
//
//    try
//    {
//        loadScene( infile.c_str(), scene );
//        scene.finalize();
//
//        OPTIX_CHECK( optixInit() ); // Need to initialize function table
//        initLaunchParams( scene );
//
//
//        if( outfile.empty() )
//        {
//            GLFWwindow* window = sutil::initUI( "Eye Renderer 3.0", width, height );
//            //glfwSetMouseButtonCallback( window, mouseButtonCallback );
//            //glfwSetCursorPosCallback  ( window, cursorPosCallback   );
//            //glfwSetWindowSizeCallback ( window, windowSizeCallback  );
//            //glfwSetKeyCallback        ( window, keyCallback         );
//            //glfwSetScrollCallback     ( window, scrollCallback      );
//            //glfwSetWindowUserPointer  ( window, &params       );
//
//            //
//            // Render loop
//            //
//            {
//                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );// Output buffer for display
//                sutil::GLDisplay gl_display;
//
//                std::chrono::duration<double> state_update_time( 0.0 );
//                std::chrono::duration<double> render_time( 0.0 );
//                std::chrono::duration<double> display_time( 0.0 );
//
//
//                char cameraInfo[100];
//                do
//                {
//                    auto t0 = std::chrono::steady_clock::now();
//                    glfwPollEvents();
//                    scene.getCamera()->moveLocally(basicController.getMovementVector());
//                    scene.getCamera()->rotateLocallyAround(basicController.getVerticalRotationAngle(),  make_float3(1.0f, 0.0f, 0.0f) );
//                    scene.getCamera()->rotateAround(basicController.getHorizontalRotationAngle(),make_float3(0.0f, 1.0f, 0.0f) );
//
//                    updateState( output_buffer, params );
//                    auto t1 = std::chrono::steady_clock::now();
//                    state_update_time += t1 - t0;
//                    t0 = t1;
//
//                    // Or maybe instead the filename should be drawn from the cameras themselves, with the compound cameras drawing from a separate file that only contains them?
//                    launchFrame( output_buffer, scene );
//                    t1 = std::chrono::steady_clock::now();
//                    render_time += t1 - t0;
//                    totalRenderTime += render_time;
//                    t0 = t1;
//
//                    displaySubframe( output_buffer, gl_display, window );
//                    t1 = std::chrono::steady_clock::now();
//                    display_time += t1 - t0;
//
//                    if(saveFlag)
//                    {
//                      sutil::ImageBuffer buffer;
//                      buffer.data = output_buffer.getHostPointer();
//                      buffer.width = output_buffer.width();
//                      buffer.height = output_buffer.height();
//                      buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
//                      sutil::displayBufferFile("output_image.ppm", buffer, false);
//                      saveFlag = false;
//                    }
//
//                    if(drawUI)
//                    {
//                      sutil::displayStats( state_update_time, render_time, display_time );
//
//                      double avg = std::chrono::duration_cast<std::chrono::milliseconds>(totalRenderTime).count()/((float)params.frame);
//                      sprintf(cameraInfo, "Camera: %i (%s)\nAvg. rendertime: %.1fms", scene.getCameraIndex(), scene.getCamera()->getCameraName(), avg);
//
//                      sutil::beginFrameImGui();
//                      sutil::displayText(cameraInfo, 10.0f, 80.0f, 250, 10);
//                      sutil::endFrameImGui();
//                    }
//
//                    glfwSwapBuffers(window);
//                }
//                while( !glfwWindowShouldClose( window ) );
//                CUDA_SYNC_CHECK();
//            }
//
//            sutil::cleanupUI( window );
//        }
//        else
//        {
//            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
//            {
//              sutil::initGLFW(); // For GL context
//              sutil::initGL();
//            }
//
//            sutil::CUDAOutputBuffer<uchar4> output_buffer_single(output_buffer_type, width, height);
//            handleCameraUpdate( params);
//            handleResize( output_buffer_single );
//            launchFrame( output_buffer_single, scene );
//
//            sutil::ImageBuffer buffer;
//            buffer.data = output_buffer_single.getHostPointer();
//            buffer.width = output_buffer_single.width();
//            buffer.height = output_buffer_single.height();
//            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
//
//            sutil::displayBufferFile(outfile.c_str(), buffer, false);
//
//            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
//            {
//                glfwTerminate();
//            }
//        }
//
//        cleanup();
//
//    }
//    catch( std::exception& e )
//    {
//        std::cerr << "Caught exception: " << e.what() << "\n";
//        return 1;
//    }
//
//    return 0;
//}
