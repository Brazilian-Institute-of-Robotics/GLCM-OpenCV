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

#ifndef GLCM_DISPATCH_HPP_
#define GLCM_DISPATCH_HPP_

#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "iostream"
#include <math.h>

class GLCMDispatch {
 public:
  void rescaleLevel(cv::InputArray input, cv::Mat output, size_t n_level);
  void grayLevelMatrix(cv::InputArray src, size_t n_level, size_t window, cv::OutputArray energy);
};

#endif  // GLCM_DISPATCH_HPP_
