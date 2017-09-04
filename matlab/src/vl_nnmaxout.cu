// @file vl_nnmaxout.cu
// @brief Maxout unit block MEX wrapper
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Based on the MatConvNet vl_nnpooling.cu function 
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#include <bits/mexutils.h>
#include <bits/datamex.hpp>
#include <bits/nnmaxout.hpp>

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose,
} ;

VLMXOption  options [] = {
  {"Verbose",         0,   opt_verbose          },
  {0,                 0,   0                    }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT=3, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int numUnits ;
  int numPieces ;
  bool backMode = false ;
  int next = IN_END ;
  mxArray const *optarg ;

  int verbosity = 0 ;
  int opt ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  if (nin < 2) { mexErrMsgTxt("The arguments are less than two.") ;}

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++verbosity ;
        break ;

      default: 
        break ;
    }
  }


  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }

  numUnits = mxGetPr(in[1])[0] ;
  numPieces = mxGetPr(in[2])[0] ;

  if (numUnits * numPieces != data.getDepth()) {
    mexErrMsgTxt("the number of hidden units is not equal to maxout layer.") ;
  }

  vl::TensorShape outputShape = vl::TensorShape(data.getHeight(),
                                                data.getWidth(), 
                                                numUnits, 
                                                data.getSize()) ;

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::DataType dataType = data.getDataType() ;
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnmaxout: mode %s; \n",  
            (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu") ;
    mexPrintf("vl_nnmaxout: numUnits: %d\n", numUnits) ;
    mexPrintf("vl_nnmaxout: numPieces: %d\n", numPieces) ;
    vl::print("vl_nnmaxout: data: ", data) ;
    vl::print("vl_nnmaxout: output: ", output) ;
  }

  
  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnmaxout_forward(context,
                                 output, data,
                                 numUnits, numPieces);
  } else {
    error = vl::nnmaxout_backward(context,
                                  derData, data, derOutput,
                                  numUnits, numPieces);
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
