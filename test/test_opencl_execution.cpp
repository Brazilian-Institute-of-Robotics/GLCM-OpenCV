/*
 * Copyright (c) 2019, SENAI Cimatec
 */

#include <gtest/gtest.h>

#include <fstream>

#include "glcm_dispatch.hpp"

struct GLCMDispatchTest : public ::testing::Test {};

TEST_F(GLCMDispatchTest, GeneralTesting) {
  GLCMDispatch glcm_dispatch;

  cv::UMat src = cv::UMat::ones(32, 32, CV_8UC1);
  src(cv::Range(0, 5), cv::Range(0, 5)) = 2;
  cv::UMat dst(cv::Size(32, 32), CV_32FC1);

  glcm_dispatch.grayLevelMatrix(src, 128, 4, dst);
  // glcm_dispatch.grayLevelMatrix(src, 128, 2, dst);

  // std::cout << dst << std::endl;

  ASSERT_TRUE(true);
}

TEST_F(GLCMDispatchTest, ImageTesting) {
  GLCMDispatch glcm_dispatch;

  cv::Mat src = cv::imread(
                     "/home/mccr/dev/resources/mccr_visual_inspection_tests/frame_set/config_2_mov_015_hor_rust_trans/"
                     "image-0000067.png");
  // cv::Mat src = cv::Mat::ones(1280,720,CV_8UC3)*256;
  if(src.empty()) throw("Shit");

  cv::UMat img_gray(src.size(), CV_8UC1);

  cv::cvtColor(src, img_gray, cv::COLOR_RGB2GRAY);
  cv::UMat src_rescale(src.size(), CV_8UC1);
  glcm_dispatch.rescaleLevel(img_gray, src_rescale.getMat(cv::ACCESS_RW), 2);
  cv::transpose(src_rescale, src_rescale);
  cv::UMat dst(src_rescale.size(), CV_32FC1);
  std::cout << dst.size() << std::endl;
  glcm_dispatch.grayLevelMatrix(src_rescale, 128, 8, dst);
  // glcm_dispatch.grayLevelMatrix(src, 128, 8, dst);
  cv::Mat dst_mat = dst.getMat(cv::ACCESS_READ);
  cv::Mat src_rescale_mat = src_rescale.getMat(cv::ACCESS_READ);
  cv::transpose(dst_mat, dst_mat);
  cv::imwrite("temp.jpg", dst_mat*256);

  // std::ofstream myfile;
  // myfile.open("output_fstream.txt");
  // myfile << cv::Mat(dst_mat, cv::Range(620, 720), cv::Range(1180, 1280));
  // myfile.close();

  ASSERT_TRUE(true);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
