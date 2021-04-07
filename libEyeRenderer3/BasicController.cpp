#include "BasicController.h"
#include <GLFW/glfw3.h>

bool BasicController::ingestKeyAction(int32_t key, int32_t action)
{
  bool output = false;
  if(action == GLFW_PRESS)
  {
      if(key == GLFW_KEY_W){
        output |= !forward;
        forward = true;
      }else if (key == GLFW_KEY_A){
        output |= !left;
        left = true;
      }else if (key == GLFW_KEY_D){
        output |= !right;
        right = true;
      }else if (key == GLFW_KEY_S){
        output |= !backward;
        backward = true;
      }else if (key == GLFW_KEY_SPACE){
        output |= !up;
        up = true;
      }else if (key == GLFW_KEY_LEFT_SHIFT){
        output |= !down;
        down = true;
      }else if (key == GLFW_KEY_UP){
        output |= !rotUp;
        rotUp = true;
      }else if (key == GLFW_KEY_DOWN){
        output |= !rotDown;
        rotDown = true;
      }else if (key == GLFW_KEY_LEFT){
        output |= !rotLeft;
        rotLeft = true;
      }else if (key == GLFW_KEY_RIGHT){
        output |= !rotRight;
        rotRight = true;
      }

  }else if(action == GLFW_RELEASE){
      if(key == GLFW_KEY_W){
        output |= forward;
        forward = false;
      }else if (key == GLFW_KEY_A){
        output |= left;
        left = false;
      }else if (key == GLFW_KEY_D){
        output |= right;
        right = false;
      }else if (key == GLFW_KEY_S){
        output |= backward;
        backward = false;
      }else if (key == GLFW_KEY_SPACE){
        output |= up;
        up = false;
      }else if (key == GLFW_KEY_LEFT_SHIFT){
        output |= down;
        down = false;
      }else if (key == GLFW_KEY_UP){
        output |= rotUp;
        rotUp = false;
      }else if (key == GLFW_KEY_DOWN){
        output |= rotDown;
        rotDown = false;
      }else if (key == GLFW_KEY_LEFT){
        output |= rotLeft;
        rotLeft = false;
      }else if (key == GLFW_KEY_RIGHT){
        output |= rotRight;
        rotRight = false;
      }
  }
  return output;
}
        //scene.getCamera()->moveLocally(make_float3(0.0f, camSpeed, 0.0f));
float3 BasicController::getMovementVector()
{
  //TODO: Could make this faster by pre-saving the multiples or something (but this is more succinct).
  float3 output = make_float3(0.0f, 0.0f, 0.0f);
  if(up)       output += speed*UP;
  if(down)     output += speed*DOWN;
  if(left)     output += speed*LEFT;
  if(right)    output += speed*RIGHT;
  if(forward)  output += speed*FORWARD;
  if(backward) output += speed*BACK;
  return output;
}

float BasicController::getVerticalRotationAngle()
{
  float out = 0.0f;
  if(rotUp)
    out += angularSpeed;
  if(rotDown)
    out -= angularSpeed;
  return out;
}

float BasicController::getHorizontalRotationAngle()
{
  float out = 0.0f;
  if(rotLeft)
    out += angularSpeed;
  if(rotRight)
    out -= angularSpeed;
  return out;
}

//float BasicController::getFocalMultiplier()
//{
//  if(zoomIn)  return (1.0f + focalSpeed);
//  if(zoomOut) return (1.0f - focalSpeed);
//  return 1.0f;
//}
