// CoreML/ANE probe binary. Loads an .mlpackage compiled from
// `gen_matmul_model.py`, dispatches it on the Neural Engine, and reports
// per-call latency.
//
// Usage:
//   coreml_matmul_probe <path/to/matmul.mlpackage> <M> <K> <N> [iters]
//
// The model has one fp16 input "x" with shape (M, K) and emits one fp16
// output of shape (M, N) representing x @ w (w baked as a constant inside
// the model). We probe ANE eligibility by requesting
// `MLComputeUnitsCPUAndNeuralEngine`; if the model can't land on ANE,
// CoreML silently falls back to CPU and we'll see CPU-class latency.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double timeMs() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch()).count();
}

int main(int argc, const char *argv[]) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <model.mlpackage> <M> <K> <N> [iters]\n",
                     argv[0]);
        return 2;
    }
    NSString *path = [NSString stringWithUTF8String:argv[1]];
    int M = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    int N = std::atoi(argv[4]);
    int iters = (argc >= 6) ? std::atoi(argv[5]) : 100;

    @autoreleasepool {
        NSError *err = nil;
        NSURL *url = [NSURL fileURLWithPath:path];
        // Compile the .mlpackage if not already compiled. CoreML caches the
        // compiled .mlmodelc form between runs in a temp directory.
        NSURL *compiled = [MLModel compileModelAtURL:url error:&err];
        if (err) {
            std::fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:cfg
                                                   error:&err];
        if (!model || err) {
            std::fprintf(stderr, "model load failed: %s\n",
                         err.localizedDescription.UTF8String);
            return 1;
        }
        // Build an fp16 input array filled with deterministic values.
        std::vector<int> shape_vec = {M, K};
        NSArray *shape = @[ @(M), @(K) ];
        MLMultiArray *x = [[MLMultiArray alloc] initWithShape:shape
                                                     dataType:MLMultiArrayDataTypeFloat16
                                                        error:&err];
        if (err) { std::fprintf(stderr, "alloc x: %s\n", err.localizedDescription.UTF8String); return 1; }
        // Fill with 1.0 (representable in fp16). The model has random W
        // baked in, so the output won't be predictable — we're only timing.
        uint16_t one_fp16 = 0x3C00;  // fp16 representation of 1.0
        for (NSInteger i = 0; i < x.count; ++i) {
            ((uint16_t *)x.dataPointer)[i] = one_fp16;
        }
        MLDictionaryFeatureProvider *inputs = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"x" : x} error:&err];
        if (err) { std::fprintf(stderr, "inputs: %s\n", err.localizedDescription.UTF8String); return 1; }

        // Warm-up — first call includes ANE program build.
        id<MLFeatureProvider> warm = [model predictionFromFeatures:inputs error:&err];
        (void)warm;
        if (err) { std::fprintf(stderr, "warmup: %s\n", err.localizedDescription.UTF8String); return 1; }

        double t0 = timeMs();
        for (int i = 0; i < iters; ++i) {
            id<MLFeatureProvider> y = [model predictionFromFeatures:inputs error:&err];
            (void)y;
            if (err) { std::fprintf(stderr, "predict %d: %s\n", i, err.localizedDescription.UTF8String); return 1; }
        }
        double t1 = timeMs();
        double avg_us = (t1 - t0) * 1000.0 / iters;
        std::printf("[coreml-ane] M=%d K=%d N=%d iters=%d avg=%.1f us/call\n",
                    M, K, N, iters, avg_us);
    }
    return 0;
}
