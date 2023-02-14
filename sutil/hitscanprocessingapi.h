//
// This is a clone of "sutilapi.h" to open up hitscan processing functions 
// under windows.

#ifndef __util_hitscanprocessingapi_h__
#define __util_hitscanprocessingapi_h__

#ifndef HITSCANAPI
#  if sutil_7_sdk_EXPORTS /* Set by CMAKE */
#    if defined( _WIN32 ) || defined( _WIN64 )
#      define HITSCANAPI __declspec(dllexport)
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux__ ) || defined ( __CYGWIN__ )
#      define HITSCANAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI HITSCANAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define HITSCANAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI HITSCANAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  else /* sutil_7_sdk_EXPORTS */

#    if defined( _WIN32 ) || defined( _WIN64 )
#      define HITSCANAPI __declspec(dllimport)
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux__ ) || defined ( __CYGWIN__ )
#      define HITSCANAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI HITSCANAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define HITSCANAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI HITSCANAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  endif /* sutil_7_sdk_EXPORTS */
#endif

#endif /* __util_hitscanprocessingapi_h__ */
