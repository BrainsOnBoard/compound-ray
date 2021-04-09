#include <iostream>

#include <sutil/sutil.h>
#include "libEyeRenderer.h"
//#include "libEyeRenderer.cpp"

// This subproject loads in libEyeRenderer and uses it to render a given scene.
// Basic controls are offered.
// It also stands as an example of how to interface the the rendering library.

int main( int argc, char* argv[] )
{
  std::cout << "Running eye Renderer GUI...\n";

  try
  {
    std::string infile = sutil::sampleDataFilePath( "ofstad-arena/ofstad-arena.gltf" );
    //loadScene( infile.c_str(), scene);
    //loadGlTFscene( infile.c_str());
    //testMe();
    setVerbosity(true);
    loadGlTFscene(infile.c_str());
    //setRenderSize(10,10);
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
