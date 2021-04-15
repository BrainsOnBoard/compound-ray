#include <iostream>

#include <sutil/sutil.h>
#include "libEyeRenderer.h"
#include <GLFW/glfw3.h>

#include <sutil/vec_math.h>
#include "BasicController.h"

// This subproject loads in libEyeRenderer and uses it to render a given scene.
// Basic controls are offered.
// It also stands as an example of how to interface the the rendering library.

bool dirtyUI = true; // a flag to keep track of if the UI has changed in any way
BasicController controller;

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
  // Handle keypresses
  if(action == GLFW_PRESS)
  {
    if(key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)
    {
      // Control keypresses
      glfwSetWindowShouldClose(window, true);
    }else{
      //// Movement keypresses

      // Camera changing
      if(key == GLFW_KEY_N)
      {
        nextCamera();
      }else if(key == GLFW_KEY_B){
        previousCamera();
      }

      dirtyUI = true;
    }
  }
  // Camera movement (mark the UI dirty if the controller has moved
  dirtyUI |= controller.ingestKeyAction(key, action);
}

static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    setRenderSize(res_x, res_y);
    dirtyUI = true;
}

int main( int argc, char* argv[] )
{
  std::cout << "Running eye Renderer GUI...\n";

  // Grab a pointer to the window
  GLFWwindow* window = (GLFWwindow*)(getWindowPointer());

  // Attach callbacks
  glfwSetKeyCallback        (window, keyCallback       );
  glfwSetWindowSizeCallback (window, windowSizeCallback);
  //glfwSetMouseButtonCallback( window, mouseButtonCallback );
  //glfwSetCursorPosCallback  ( window, cursorPosCallback   );
  //glfwSetWindowSizeCallback ( window, windowSizeCallback  );
  //glfwSetScrollCallback     ( window, scrollCallback      );
  //glfwSetWindowUserPointer  ( window, &params       );
  std::string infile = sutil::sampleDataFilePath( "ofstad-arena/ofstad-arena.gltf" );

  try
  {
    // Turn off verbose logging
    setVerbosity(false);

    // Load the file
    std::cout << "Loading file \"" << infile << "\"..." << std::endl;
    loadGlTFscene(infile.c_str());

    // The main loop
    do
    {
      glfwPollEvents(); // Check if anything's happened, user-input-wise.

      if(controller.isActivelyMoving())
      {
        float3 t = controller.getMovementVector();// Local translation
        translateCameraLocally(t.x, t.y, t.z);
        float va = controller.getVerticalRotationAngle();
        float vh = controller.getHorizontalRotationAngle();
        rotateCameraLocallyAround(va, 1.0f, 0.0f, 0.0f);
        rotateCameraAround(vh, 0.0f, 1.0f, 0.0f);
        dirtyUI = true;
      }

      // Render and display the frame if anything's changed (movement or frame)
      if(dirtyUI)
      {

        renderFrame();
        displayFrame();
        dirtyUI = false; // Comment this out to force constant re-rendering
      }

    }while( !glfwWindowShouldClose( window ) );
    stop();
  }
  catch(std::exception& e)
  {
    std::cerr << "Caught exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

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
