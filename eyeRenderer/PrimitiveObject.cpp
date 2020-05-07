#include "PrimitiveObject.h"

float3 PrimitiveObject::X_VECTOR = make_float3(1.0f, 0.0f, 0.0f);
float3 PrimitiveObject::Y_VECTOR = make_float3(0.0f, 1.0f, 0.0f);
float3 PrimitiveObject::Z_VECTOR = make_float3(0.0f, 0.0f, 1.0f);

PrimitiveObject::PrimitiveObject(void)
{
  dirty = true;
}
PrimitiveObject::~PrimitiveObject(void){}

void PrimitiveObject::recalculateIfDirty()
{
  if(dirty)
    recalculateProperties();
}

OptixAabb* PrimitiveObject::getBoundsPointer()
{
  return &bounds;
}
