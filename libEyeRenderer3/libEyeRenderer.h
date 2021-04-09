#ifndef LIB_EYE_RENDERER_3_H
#define LIB_EYE_RENDERER_3_H
#include <cstddef>

extern "C"
{
  // Configuration
  void setVerbosity(bool v);                // turns on/off the '[PyEye]' debug outputs
  void loadGlTFscene(const char* filepath); // Loads a given gltf file
  void stop(void);                          // Stops the eyeRenderer in a slightly more elegant way
  void setRenderSize(int w, int h);         // Sets the width an height of the rendering frame
  double renderFrame(void);                 // Actually renders the frame, returns the time it took to render the frame (in ms)
  void displayFrame(void);                  // Displays the rendered frame to the open GL display window
  void saveFrameAs(char* ppmFilename);      // Saves the frame (must be a .ppm filename)
  unsigned char* getFramePointer(void);     // Retrieves a pointer to the frame data

  // Camera control
  size_t getCameraCount(void);
  void nextCamera(void);
  void previousCamera(void);
  size_t getCurrentCameraIndex(void);
  const char* getCurrentCameraName();
  void gotoCamera(int index);
  bool gotoCameraByName(char* name);
  //void setCameraPosition(float x, float y, float z);
  //float* getCameraPosition();
  //void pointCameraAt
  //void setCameraLocalSpace
  //void rotateCameraAround
  //void rotateCameraLocallyAround
  //void moveCamera
  //void moveCameraLocally

  // Compound-specific
  //int getOmmatidialCameraCount(void);
  bool isCompoundEyeActive(void);
  void setCurrentEyeSamplesPerOmmatidium(int s);// Changes the current eye samples per ommatidium. WARNING: This resets the random seed values. A render must be called to regenerate them, this will take significantly longer than a frame render.
  int  getCurrentEyeSamplesPerOmmatidium(void);// Returns the current eye samples per ommatidium
  // size_t getOmmatidialCount(void);
  // void setOmmatidia(size_t count, Ommatidium* omms); // Sets the ommatidia for the eye
}

#endif
