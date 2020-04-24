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

#include <cuda_runtime.h>
#include <optixDemandTexture.h>
#include <sutil/vec_math.h>

// Whether to use tex2DLod or tex2DGrad
//#define USE_TEX2DLOD 1

extern "C" {
__constant__ Params params;
}

//------------------------------------------------------------------------------
//
// Per ray data for closets hit program and functions to access it
//
//------------------------------------------------------------------------------

struct RayPayload
{
    // Return value
    float3 rgb;

    // Ray differential
    float3 origin_dx;
    float3 origin_dy;
    float3 direction_dx;
    float3 direction_dy;

    // padding
    int32_t  pad;
};


static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RayPayload* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<RayPayload*>( unpackPointer( u0, u1 ) );
}


//------------------------------------------------------------------------------
//
// Determine the image pixel to render based on the sample index for mulit-gpu
//
//------------------------------------------------------------------------------

static const int32_t TILE_WIDTH  = 8;
static const int32_t TILE_HEIGHT = 4;

static __forceinline__ __device__ uint2 getWorkIndex( int32_t gpu_idx, int32_t sample_idx, int32_t width, int32_t height, int32_t num_gpus )
{
    const int tile_strip_width    = TILE_WIDTH * num_gpus;
    const int tile_strip_height   = TILE_HEIGHT;
    const int num_tile_strip_cols = width / tile_strip_width + ( width % tile_strip_width == 0 ? 0 : 1 );

    const int tile_strip_idx     = sample_idx / ( TILE_WIDTH * TILE_HEIGHT );
    const int tile_strip_y       = tile_strip_idx / num_tile_strip_cols;
    const int tile_strip_x       = tile_strip_idx - tile_strip_y * num_tile_strip_cols;
    const int tile_strip_x_start = tile_strip_x * tile_strip_width;
    const int tile_strip_y_start = tile_strip_y * tile_strip_height;

    const int tile_pixel_idx = sample_idx - ( tile_strip_idx * TILE_WIDTH * TILE_HEIGHT );
    const int tile_pixel_y   = tile_pixel_idx / TILE_WIDTH;
    const int tile_pixel_x   = tile_pixel_idx - tile_pixel_y * TILE_WIDTH;

    const int tile_offset_x = ( gpu_idx + tile_strip_y % num_gpus ) % num_gpus * TILE_WIDTH;

    const int pixel_y = tile_strip_y_start + tile_pixel_y;
    const int pixel_x = tile_strip_x_start + tile_pixel_x + tile_offset_x;
    return make_uint2( pixel_x, pixel_y );
}


//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

// trace a ray
static __forceinline__ __device__ void trace( OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax, RayPayload* prd )
{
    uint32_t u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,     // SBT offset
            RAY_TYPE_COUNT,        // SBT stride
            RAY_TYPE_RADIANCE,     // missSBTIndex
            u0, u1 );
}


// make a uchar color from a float3
__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    return make_uchar4( static_cast<uint8_t>( clamp( c.x, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<uint8_t>( clamp( c.y, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<uint8_t>( clamp( c.z, 0.0f, 1.0f ) * 255.0f ), 255u );
}


// Convert Cartesian coordinates to polar coordinates
__forceinline__ __device__ float3 cartesian_to_polar( const float3& v )
{
    float azimuth;
    float elevation;
    float radius = length( v );

    float r = sqrtf( v.x * v.x + v.y * v.y );
    if( r > 0.0f )
    {
        azimuth   = atanf( v.y / v.x );
        elevation = atanf( v.z / r );

        if( v.x < 0.0f )
            azimuth += M_PI;
        else if( v.y < 0.0f )
            azimuth += M_PI * 2.0f;
    }
    else
    {
        azimuth = 0.0f;

        if( v.z > 0.0f )
            elevation = +M_PI_2;
        else
            elevation = -M_PI_2;
    }

    return make_float3( azimuth, elevation, radius );
}


//------------------------------------------------------------------------------
//
// Functions related to demand textures
//
//------------------------------------------------------------------------------

// Check whether a specified miplevel of a demand-loaded texture is resident, recording a request if not.
inline __device__ void requestMipLevel( unsigned int textureId, const DemandTextureSampler& sampler, unsigned int mipLevel, bool& isResident )
{
    // A page id consists of the texture id (upper 28 bits) and the miplevel number (lower 4 bits).
    const unsigned int requestedPage = textureId << 4 | mipLevel;

    // The paging context was provided as a launch parameter.
    const OptixPagingContext& context = params.pagingContext;

    // Check whether the requested page is resident, recording a request if not.
    optixPagingMapOrRequest( context.usageBits, context.residenceBits, context.pageTable, requestedPage, &isResident );
}


// Request the demand-loaded mip levels that might be used in a tex2DLod call using the given lod
inline __device__ void requestLod( unsigned int textureId, const DemandTextureSampler& sampler, float lod, bool& isResident )
{
    // The software calculation of the MIP level is not exactly the same as the hardware, so conservatively
    // load extra MIP levels
    const float MIP_REQUEST_OFFSET = 0.2f;

    const int coarsestMiplevel = sampler.totalMipLevels-1;
    const int lowerLevel       = clamp( static_cast<int>( lod - MIP_REQUEST_OFFSET ),     0, coarsestMiplevel );
    const int upperLevel       = clamp( static_cast<int>( lod + MIP_REQUEST_OFFSET + 1 ), 0, coarsestMiplevel );

    requestMipLevel( textureId, sampler, lowerLevel, isResident );
    if ( (lowerLevel + 1) < upperLevel )
    {
        bool isResident2 = true;
        requestMipLevel( textureId, sampler, lowerLevel+1, isResident2 );
        isResident = isResident && isResident2;
    }
    if( upperLevel != lowerLevel )
    {
        bool isResident3 = true;
        requestMipLevel( textureId, sampler, upperLevel, isResident3 );
        isResident = isResident && isResident3;
    }
}


// Calculate lod that would be used by a tex2DGrad call
inline __device__ float getLodFromTexDerivatives( const float2& ddx, const float2& ddy, float totalMipLevels, float invAnisotropy )
{
    const float MIN_FILTERWIDTH = 0.0000001f;

    float dx = length( ddx );
    float dy = length( ddy );

    // Use the smaller of the two lengths as the filter width, up to the max anisotropy limit
    float filterWidth = ( dx > dy ) ? fmax( dy, dx * invAnisotropy ) : fmax( dx, dy * invAnisotropy );
    filterWidth       = fmax( MIN_FILTERWIDTH, filterWidth );

    float lod = totalMipLevels + log2( filterWidth ) - 1.0f;
    return lod;
}


// Fetch from a demand-loaded texture with a specified LOD (using tex2DLod).  The necessary mip levels
// are requested if they are not resident, which is indicated by the boolean result parameter.
inline __device__ float4 tex2DLodLoadOrRequest( unsigned int textureId, const DemandTextureSampler& sampler,
                                                float x, float y, float mipLevel, bool& isResident )
{
    requestLod( textureId, sampler, mipLevel, isResident );

    if( isResident )
        return tex2DLod<float4>( sampler.texture, x, y, mipLevel );
    else
        return make_float4( 1.f, 0.f, 1.f, 0.f );
}


// Fetch from a demand-loaded texture with the given texture derivatives (using tex2DGrad). As above,
// the necessary mip levels are requested if they are not present.
inline __device__ float4 tex2DGradLoadOrRequest( unsigned int textureId, const DemandTextureSampler& sampler,
                                                 float x, float y, const float2& ddx, const float2& ddy, bool& isResident )
{
    float lod = getLodFromTexDerivatives( ddx, ddy, sampler.totalMipLevels, INV_ANISOTROPY );
    requestLod( textureId, sampler, lod, isResident );

    if( isResident )
        return tex2DGrad<float4>( sampler.texture, x, y, ddx, ddy );
    else
        return make_float4( 1.f, 0.f, 1.f, 0.f );
}


// Compute texture-space texture derivatives (ddx and ddy) based on ray differentials and world-space
// texture derivatives.
inline __device__ void computeTextureDerivatives( const RayPayload* prd,   // ray data, including ray differentials
                                                  const float3&     dPds,  // world space s texture derivative
                                                  const float3&     dPdt,  // world space t texture derivative
                                                  const float3&     dir,   // ray direction normal
                                                  const float       thit,  // distance to intersection
                                                  const float3&     N,     // geometric normal
                                                  float2&           ddx,   // texture derivatives (out)
                                                  float2&           ddy )
{
    // Compute the ray differential values at the intersection point
    float3 rdx = prd->origin_dx + thit * prd->direction_dx;
    float3 rdy = prd->origin_dy + thit * prd->direction_dy;

    // Project the ray differentials the plane of the rdx and rdy.
    // Since rdx and rdy are tangents, the surface normal is the plane normal.
    float tx = dot( rdx, N ) / dot( dir, N );
    float ty = dot( rdy, N ) / dot( dir, N );
    rdx -= tx * dir;
    rdy -= ty * dir;

    // Compute the texture derivatives in texture space. These are
    // 1. proportional to the length of the projected differentials
    // 2. proportional to the cosine between the texture derivatives and the projected ray differentials
    // 3. inversely proportional to length of the texture derivatives
    ddx.x = dot( dPds, rdx ) / dot( dPds, dPds );
    ddx.y = dot( dPdt, rdx ) / dot( dPdt, dPdt );

    ddy.x = dot( dPds, rdy ) / dot( dPds, dPds );
    ddy.y = dot( dPdt, rdy ) / dot( dPdt, dPdt );
}


//------------------------------------------------------------------------------
//
// Optix programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixelIdx = launch_idx.x * params.image_width + launch_idx.y;
    const uint2 idx = getWorkIndex( params.device_idx, pixelIdx, params.image_width, params.image_height, params.num_devices );
    const float imageWidth  = params.image_width;
    const float imageHeight = params.image_height;

    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float2 d = 2.0f * make_float2( static_cast<float>( idx.x ) / imageWidth, static_cast<float>( idx.y ) / imageHeight ) - 1.0f;

    const float3 origin    = params.eye;
    const float3 direction = normalize( d.x * U + d.y * V + W );

    RayPayload prd;
    prd.rgb          = make_float3( 0.0f );
    prd.origin_dx    = make_float3( 0.0f );
    prd.origin_dy    = make_float3( 0.0f );
    const float Wlen = length( W );
    // TODO: This is not 100% correct, since U and V are not perpendicular to the ray direction
    prd.direction_dx = U * ( 2.0f / ( imageWidth * Wlen ) );
    prd.direction_dy = V * ( 2.0f / ( imageHeight * Wlen ) );

    trace( params.handle, origin, direction,
           0.00f,  // tmin
           1e16f,  // tmax
           &prd );

    params.result_buffer[idx.y * params.image_width + idx.x] = make_color( prd.rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData*   rt_data = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RayPayload* prd     = getPRD();

    prd->rgb = make_float3( rt_data->r, rt_data->g, rt_data->b );
}


extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3  orig    = optixGetObjectRayOrigin();
    const float3  dir     = optixGetObjectRayDirection();

    const float3 center = {0.f, 0.f, 0.f};
    const float  radius = hg_data->radius;
    const float3 O      = orig - center;
    const float  l      = 1 / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - radius * radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float  root11         = 0.0f;
        const float3 shading_normal = ( O + ( root1 + root11 ) * D ) / radius;

        float3 polar    = cartesian_to_polar( shading_normal );
        float3 texcoord = make_float3( polar.x * 0.5f * M_1_PI, ( polar.y + M_PI_2 ) * M_1_PI, polar.z / radius );

        unsigned int p0, p1, p2;
        p0 = float_as_int( texcoord.x );
        p1 = float_as_int( texcoord.y );
        p2 = float_as_int( texcoord.z );

        unsigned int n0, n1, n2;
        n0 = float_as_int( shading_normal.x );
        n1 = float_as_int( shading_normal.y );
        n2 = float_as_int( shading_normal.z );

        optixReportIntersection( root1,         // t hit
                                 0,             // user hit kind
                                 p0, p1, p2,    // texture coordinates
                                 n0, n1, n2 );  // geometric normal
    }
}


extern "C" __global__ void __closesthit__ch()
{
    bool isResident;

    // The demand-loaded texture id is provided in the hit group data.  It's used as an index into
    // the sampler array, which is a launch parameter.
    HitGroupData*               hg_data      = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    unsigned int                textureId    = hg_data->demand_texture_id;
    const float                 textureScale = hg_data->texture_scale;
    const DemandTextureSampler& sampler      = params.demandTextures[textureId];

    // The texture coordinates and normal are calculated by the intersection shader are provided as attributes.
    const float3 texcoord = make_float3( int_as_float( optixGetAttribute_0() ), int_as_float( optixGetAttribute_1() ),
                                         int_as_float( optixGetAttribute_2() ) );

    const float3 N = make_float3( int_as_float( optixGetAttribute_3() ), int_as_float( optixGetAttribute_4() ),
                                  int_as_float( optixGetAttribute_5() ) );

    // Compute world space texture derivatives based on normal and radius
    float radius = hg_data->radius;
    float3 dPds = radius * 2.0f * M_PI * make_float3( N.y, -N.x, 0.0f );
    float3 dPdt = radius * M_PI * normalize( cross( N, dPds ) );

    // Compute final texture coordinates
    float s = texcoord.x * textureScale;
    float t = (1.0f - texcoord.y) * textureScale;

    // Get texture space texture derivatives based on ray differentials
    float2 ddx, ddy;
    RayPayload* prd = getPRD();
    const float3 dir = optixGetWorldRayDirection();
    const float thit = optixGetRayTmax();
    computeTextureDerivatives( prd, dPds, dPdt, dir, thit, N, ddx, ddy );
    float biasScale = powf( 2.0f, params.mipLevelBias );
    ddx *= textureScale * biasScale;
    ddy *= textureScale * biasScale;

#ifdef USE_TEX2DLOD
    // Sample texture using tex2DLod, using texture derivatives to determine mip level
    // Note that passing 1.0f as the invAnisotropy to getLodFromTexDerivatives
    // will choose the lod based on the larger of the two derivatives, rather than the smaller.
    //float lod = getLodFromTexDerivatives( ddx, ddy, sampler.totalMipLevels, 1.0f );
    float lod = getLodFromTexDerivatives( ddx, ddy, sampler.totalMipLevels, 1.0f );
    float4 pixel = tex2DLodLoadOrRequest( textureId, sampler, s, t, lod, isResident );
#else
    // Sample texture using tex2DGrad
    float4 pixel = tex2DGradLoadOrRequest( textureId, sampler, s, t, ddx, ddy, isResident );
#endif

    prd->rgb = make_float3(pixel);
}
