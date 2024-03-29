//
// Created by Vallabh Ramakanth on 24/10/18.
//

#ifndef V4_OCV_COMMON_HPP
#define V4_OCV_COMMON_HPP

#endif //V4_OCV_COMMON_HPP

#include <android/log.h>
#define LOG_TAG "JNI_COMMON"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#include <time.h> // clock_gettime

static inline int64_t getTimeMs()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000 + now.tv_nsec/1000000;
}

static inline int getTimeInterval(int64_t startTime)
{
    return int(getTimeMs() - startTime);
}