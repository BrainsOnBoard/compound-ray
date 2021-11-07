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
      }else if(key == GLFW_KEY_PAGE_UP){
        changeCurrentEyeSamplesPerOmmatidiumBy(10);
      }else if(key == GLFW_KEY_PAGE_DOWN){
        changeCurrentEyeSamplesPerOmmatidiumBy(-10);
      }else if(key = GLFW_KEY_C){
        saveFrameAs("output.ppm");
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

void printHelp()
{
  std::cout << "USAGE:\nnewGuiEyeRenderer -f <path to gltf scene>" << std::endl << std::endl;
  std::cout << "\t-h\tDisplay this help information." << std::endl;
  std::cout << "\t-f\tPath to a gltf scene file (absolute or relative to data folder, e.g. 'natural-standin-sky.gltf')." << std::endl;
}

int main( int argc, char* argv[] )
{
  std::cout << "Running eye Renderer GUI...\n";

  // Parse Inputs
  std::string path = "";
  for (int i=0; i<argc; i++)
  {
    std::string arg = std::string(argv[i]);
    if(arg == "-h")
    {
      printHelp();
      return 0;
    }
    else if(arg == "-f")
    {
      i++;
      path = std::string(argv[i]);
    }
  }

  if(path == "")
  {
    printHelp();
    return 1;
  }

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
  std::string infile = sutil::sampleDataFilePath(path.c_str());

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

      // Render and display the frame if anything's changed (movement or window resize etc)
      // also re-render the frame if the current camera is a compound eye in order to get a
      // better feeling of the stochastic spread encountered.
      if(dirtyUI || isCompoundEyeActive())
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
