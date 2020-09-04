/*
 * Copyright (c) 2020, SENAI CIMATEC
 */

#include <opencv2/core/ocl.hpp>

#include "benchmark/benchmark.h"
#include "glcm_dispatch.hpp"

class GLCMBenchmarkFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) {}

  void TearDown(const ::benchmark::State& state) {}
};

BENCHMARK_DEFINE_F(GLCMBenchmarkFixture, OpeclDispatchBM)(benchmark::State& st) {  // NOLINT(runtime/references)
  GLCMDispatch glcm_dispatch;

  cv::UMat src = cv::UMat::ones(1280, 720, CV_8UC1);
  cv::UMat dst(cv::Size(1280, 720), CV_32FC1);

  for (auto _ : st) {
    glcm_dispatch.grayLevelMatrix(src, 128, 8, dst);
  }
}

BENCHMARK_REGISTER_F(GLCMBenchmarkFixture, OpeclDispatchBM)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
