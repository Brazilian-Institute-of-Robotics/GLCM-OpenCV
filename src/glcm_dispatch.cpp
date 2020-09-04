/*=================================================================
 * Calculate GLCM(Gray-level Co-occurrence Matrix) By OpenCV.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/

#include "glcm_dispatch.hpp"

#include "CL/cl.h"
#include "opencl_kernels_glcm.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"

void GLCMDispatch::rescaleLevel(cv::InputArray input, cv::Mat output, size_t n_level) {
  cv::UMat tmp;
  input.copyTo(tmp);
  if (tmp.channels() == 3) cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
  cv::Mat dst = output;
  cv::Mat tmp_mat = tmp.getMat(cv::ACCESS_READ);

  // // 直方图均衡化
  // // Equalize Histogram
  // // equalizeHist(tmp, tmp);

  for (int j = 0; j < tmp.rows; j++) {
    const uchar* current = tmp_mat.ptr<uchar>(j);
    uchar* output_char = output.ptr<uchar>(j);

    for (int i = 0; i < tmp.cols; i++) {
      output.at<uchar>(j, i) = cv::saturate_cast<uchar>(tmp_mat.at<uchar>(j, i) / 2);
    }
  }
}

void GLCMDispatch::grayLevelMatrix(cv::InputArray src, size_t n_level, size_t window, cv::OutputArray energy) {
  double min_val, max_val;
  cv::minMaxIdx(src, &min_val, &max_val);
  CV_Assert(n_level > max_val);
  cv::ocl::Device dev = cv::ocl::Device::getDefault();
  bool doubleSupport = dev.doubleFPConfig() > 0;
  int stype = src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype), ddepth = energy.depth();
  int rowsPerWI = dev.isIntel() ? 4 : 1;
  // int kercn = cv::ocl::predictOptimalVectorWidthMax(src, energy);
  int kercn = 1;

  size_t globalsize[2] = {(size_t)src.cols() * cn / kercn, ((size_t)src.rows() + rowsPerWI - 1) / rowsPerWI};
  // if (((globalsize[0]) % (window)) != 0 || ((globalsize[1]) % (window)) != 0) {
  //   globalsize[0] = src.cols();
  //   globalsize[1] = src.rows();
  // }

  int level_cn = 1;

  globalsize[0] = globalsize[0] * window;
  globalsize[1] = globalsize[1] * window / level_cn;

  size_t localsize[2] = {window, window / level_cn};

  if ((n_level % window) != 0) throw(std::invalid_argument("Cannot do this level window combination"));
  int gray_levels_per_local = n_level / window * n_level / window;

  char cvt[40];

  cv::ocl::Kernel k;

  bool was_created =
      k.create("glcm", cv::ocl::glcm::glcm_oclsrc,
               cv::format("-D %s -D srcT1=%s -D cn=%d -D dstT1=%s%s -D rowsPerWI=%d -D convertToDT=%s "
                          "-D n_level=%zu -D window=%zu -D grayLevelsPerLocal=%d",
                          "NO_USE_GRAYLEVEL", cv::ocl::typeToStr(sdepth), level_cn, cv::ocl::typeToStr(ddepth),
                          doubleSupport ? " -D DOUBLE_SUPPORT" : "", rowsPerWI,
                          cv::ocl::convertTypeStr(sdepth, ddepth, 1, cvt), n_level, window, gray_levels_per_local));

  if (!was_created) {
    throw(std::invalid_argument("Could not create kernel"));
  }

  cv::UMat u_src = src.getUMat();
  cv::UMat u_src_brd;
  /* copyMakeBorder adds f(windows)ms - 0.2ms for window = 16 - on the function execution */
  cv::copyMakeBorder(src, u_src_brd, 0, window*2, 0, window*2, cv::BORDER_REPLICATE);

  cv::UMat u_energy = energy.getUMat();

  cv::ocl::KernelArg srcarg = cv::ocl::KernelArg::ReadOnlyNoSize(u_src_brd),
                     dstarg = cv::ocl::KernelArg::ReadWrite(u_energy, cn, kercn);

  int argidx = k.set(0, srcarg);
  argidx = k.set(argidx, dstarg);

  // std::cout << dev.maxWorkGroupSize() << std::endl;

  // std::cout << "rowsPerWI: " << rowsPerWI << std::endl;
  // std::cout << "kercn: " << kercn << std::endl;
  // std::cout << "cn: " << cn << std::endl;
  // std::cout << "globalsize[0]: " << globalsize[0] << std::endl;
  // std::cout << "globalsize[1]: " << globalsize[1] << std::endl;

  k.runProfiling(2, globalsize, localsize);
}

// int main() {
//   cv::UMat src = cv::UMat::ones(100, 100, CV_8UC1);
//   cv::UMat src2 = cv::UMat::ones(100, 100, CV_8UC1);
//   cv::UMat dst(cv::Size(100, 100), CV_8UC1);
//   cv::UMat mask;
//   cv::ocl::Device dev = cv::ocl::Device::getDefault();
//   bool doubleSupport = dev.doubleFPConfig() > 0;
//   int stype = src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype), ddepth = dst.depth();
//   int rowsPerWI = dev.isIntel() ? 4 : 1;
//   int kercn = cv::ocl::predictOptimalVectorWidthMax(src, src2, dst);
//   char cvt[40];

//   cv::ocl::Kernel k;
//   bool create = k.create("glcm", cv::ocl::glcm::glcm_oclsrc,
//                          cv::format("-D %s -D srcT1=%s -D cn=%d -D dstT1=%s%s -D rowsPerWI=%d -D convertToDT=%s",
//                                     "ACCUMULATE", cv::ocl::typeToStr(sdepth), kercn, cv::ocl::typeToStr(ddepth),
//                                     doubleSupport ? " -D DOUBLE_SUPPORT" : "", rowsPerWI,
//                                     cv::ocl::convertTypeStr(sdepth, ddepth, 1, cvt)));

//   std::cout << "create? " << create << std::endl;

//   cv::ocl::KernelArg srcarg = cv::ocl::KernelArg::ReadOnlyNoSize(src),
//                      src2arg = cv::ocl::KernelArg::ReadOnlyNoSize(src2),
//                      dstarg = cv::ocl::KernelArg::ReadWrite(dst, cn, kercn),
//                      maskarg = cv::ocl::KernelArg::ReadOnlyNoSize(mask);

//   int argidx = k.set(0, srcarg);
//   argidx = k.set(argidx, dstarg);

//   std::cout << "argidx " << argidx << std::endl;

//   size_t globalsize[2] = {(size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI};

//   std::cout << "CN: " << ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI << std::endl;

//   // k.run(2, globalsize, NULL, true);

//   std::cout << k.runProfiling(2, globalsize, NULL) << std::endl;
//   // cv::accumulate(src, dst);
//   // cv::accumulate(src, dst);

//   std::cout << dst << std::endl;

//   return 0;
// }
