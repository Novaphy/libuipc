#pragma once
// Override Corex host_defines noinline macro for libstdc++ compatibility.
// Keep all original definitions, then patch __noinline__ to the plain
// attribute token expected by headers that use __attribute__((__noinline__)).
#include_next <crt/host_defines.h>

#ifdef __noinline__
#undef __noinline__
#endif
#define __noinline__ noinline
