#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <CoreGraphics/CoreGraphics.h>
#include <QuartzCore/QuartzCore.h>
#include <cstdio>

int main() {
    @autoreleasepool {
        printf("os: %s\n", [[NSProcessInfo processInfo].operatingSystemVersionString UTF8String]);
#if defined(__arm64__)
        printf("arch: arm64\n");
#elif defined(__x86_64__)
        printf("arch: x86_64\n");
#else
        printf("arch: unknown\n");
#endif
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "metal_available: false (MTLCreateSystemDefaultDevice returned nil)\n");
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            fprintf(stderr, "MTLCopyAllDevices count: %lu\n", (unsigned long)all.count);
            return 1;
        }
        printf("metal_available: true\n");
        printf("metal_device: %s\n", device.name.UTF8String);
    }
    return 0;
}
