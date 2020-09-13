#pragma once 

#include "cameras/GenericCamera.h"

class BasicController {

  public:
    float speed = 0.02f;
    float angularSpeed = M_PIf * 0.5f/180;
    //float focalSpeed = 0.05f;

    bool ingestKeyAction(int32_t key, int32_t action);// Returns true if an update is made
    float3 getMovementVector();
    float getVerticalRotationAngle();
    float getHorizontalRotationAngle();// Rightward is positive
    float getFocalMultiplier();
    // TODO: Make the getMovementVector function take a timestep to disconnect it from framerate
    //       Also to combine the rotation methods.

  private:
    static constexpr float3 UP      = {0.0f,  1.0f, 0.0f};
    static constexpr float3 DOWN    = {0.0f, -1.0f, 0.0f};
    static constexpr float3 LEFT    = {-1.0f, 0.0f, 0.0f};
    static constexpr float3 RIGHT   = { 1.0f, 0.0f, 0.0f};
    static constexpr float3 FORWARD = {0.0f, 0.0f,  1.0f};
    static constexpr float3 BACK    = {0.0f, 0.0f, -1.0f};

    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool rotUp = false;
    bool rotDown = false;
    bool rotLeft = false;
    bool rotRight = false;
    bool zoomIn = false;
    bool zoomOut = false;
};
