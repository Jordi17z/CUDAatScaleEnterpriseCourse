/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <nppi.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

const char *kFilterTypeDefault = "gaussian";

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

void SetupImageProcessing(const std::string & s_filename, npp::ImageCPU_8u_C1 &o_host_src, npp::ImageNPP_8u_C1 & o_device_src, npp::ImageNPP_8u_C1 &o_device_dst)
{

  //Load image
  npp::loadImage(s_filename, o_host_src);
  o_device_src = npp::ImageNPP_8u_C1(o_host_src); // Copy to device

  //Create the output image
  NppiSize o_size_ROI = {static_cast<int> (o_device_src.width()), static_cast<int>(o_device_src.height())};
  o_device_dst = npp::ImageNPP_8u_C1(o_size_ROI.width, o_size_ROI.height);
}


void ApplyGaussianFilter(const npp::ImageNPP_8u_C1 & o_device_src, npp::ImageNPP_8u_C1 & o_device_dst)
{

  NppiSize o_size_ROI = {static_cast<int>(o_device_src.width()), static_cast<int> (o_device_src.height())};

  NppiSize o_mask = {5, 5};
  NppiMaskSize e_mask_size(NPP_MASK_SIZE_5_X_5);
  
  //Apply the bilateral filter:
  NPP_CHECK_NPP(nppiFilterGauss_8u_C1R(
      o_device_src.data(), o_device_src.pitch(),
      o_device_dst.data(), o_device_dst.pitch(),
      o_size_ROI,
      e_mask_size));

  
}

void ApplyLaplaceFilter(const npp::ImageNPP_8u_C1 &o_device_src, npp::ImageNPP_8u_C1 &o_device_dst)
{
  NppiSize o_size_ROI = {(int)o_device_src.width(),(int)o_device_src.height()};

  NppiSize o_src_size = {(int)o_device_src.width(), (int)o_device_src.height()};
  NppiPoint o_src_offset = {0, 0};

  NppiSize o_mask = {5, 5};
  NppiPoint o_anchor = {o_mask.width / 2, o_mask.height / 2};

  NppiMaskSize e_mask_size(NPP_MASK_SIZE_5_X_5);

  //Apply the affine transformation
  NPP_CHECK_NPP(nppiFilterLaplaceBorder_8u_C1R(
    o_device_dst.data(), o_device_dst.pitch(),
    o_src_size,
    o_src_offset,
    o_device_dst.data(), o_device_dst.pitch(), 
    o_size_ROI, e_mask_size,
    NPP_BORDER_REPLICATE
  ));

}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  std::string s_filename;
  std::string s_result_filename = "./data/output_image.pgm";
  char *filter_type;
  char *file_name;
  char *file_path;

  //Parse command line arguments
  if (checkCmdLineFlag(argc, (const char **)argv, "input"))
  {
    getCmdLineArgumentString(argc, (const char **) argv, "input", &file_name);  //Gets the file name
    file_path = sdkFindFilePath(file_name, argv[0]);                            //Searches the file in our directories
  }
  
  if (checkCmdLineFlag(argc, (const char **)argv, "-f"))
  {
    getCmdLineArgumentString(argc, (const char **)argv, "-f", &filter_type);
  }else{
    filter_type = (char *)kFilterTypeDefault;
  }

  if(file_path)
    s_filename = file_path;
  

  printf("Filename Value: %s\n", s_filename.c_str());
  printf("Filter Type Value: %s\n", filter_type);

  try{

    findCudaDevice(argc, (const char **)argv); // Initialize CUDA device

    //Setup the image
    npp::ImageCPU_8u_C1 o_host_src;
    npp::ImageNPP_8u_C1 o_device_src, o_device_dst;

    SetupImageProcessing(s_filename, o_host_src, o_device_src, o_device_dst);
    

    //Apply depeding on what filter:
    if( strcmp(filter_type, "gaussian") == 0){
      ApplyGaussianFilter(o_device_src, o_device_dst);

    } else if ( strcmp(filter_type, "laplace") == 0){
      ApplyLaplaceFilter(o_device_src, o_device_dst);
    
    }else{
      fprintf(stderr, "Unknown filter type: %s\n", filter_type);
      exit(EXIT_FAILURE);
    }

    
    // Copying result from device to host
    npp::ImageCPU_8u_C1 o_host_dst(o_device_dst.size());
    o_device_dst.copyTo(o_host_dst.data(), o_host_dst.pitch());     //Error here with gaussian filter
  
    // Save the output image
    saveImage(s_result_filename, o_host_dst);
    fprintf(stdout, "Saved image: %s\n", s_result_filename.c_str());

    nppiFree(o_device_dst.data());
    nppiFree(o_device_src.data());

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
