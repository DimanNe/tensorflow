#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tfu {

    tensorflow::Input InputFromTensorShape(const tensorflow::TensorShape &Shape);


    /// tensorflow::Output - represents a tensor value produced by an Operation.
    /// Usage example:
    ///    to::Variable Result = to::Variable(s, {1, 2}, tf::DT_DOUBLE);
    ///    to::Relu Relu = to::Relu(s, Result);
    ///    tf::TensorShape BeforeRelu = tfu::GetShapeOfIthInput(Relu, 0);
    ///    LOG(INFO) << BeforeRelu.DebugString();
    /// will output:
    /// [1,2]
    // tensorflow::TensorShape GetShapeOfIthInput(const tensorflow::Output &Output, size_t i);

    tensorflow::PartialTensorShape GetShapeOfOutput(const tensorflow::Output &Output);

}
