/***
Copyright 2012 DEVSIM LLC

SPDX-License-Identifier: Apache-2.0
***/
#ifndef SYMDIFF_IMPORT_HH
#define SYMDIFF_IMPORT_HH
#if defined _WIN32
#define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
#define DLL_LOCAL __declspec(dllimport)
#else
#define DLL_PUBLIC __attribute__ ((visibility("default")))
#define DLL_LOCAL  __attribute__ ((visibility("hidden")))
#endif
#endif

#if defined _WIN32
#define DLL_PROTECTED DLL_PUBLIC
#elif defined(SYMDIFF_NO_EXTERN_SYMBOLS)
#define DLL_PROTECTED DLL_LOCAL
#else
#define DLL_PROTECTED DLL_PUBLIC
#endif

