#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tfu {

    tensorflow::Input InputFromTensorShape(const tensorflow::TensorShape &Shape);

    class TVariableInitializer {
    public:
        template <class TDistr, class... TDistrArgs>
        tensorflow::ops::Variable Create(const tensorflow::Scope &               Scope,
                                         const tensorflow::TensorShape &         Shape,
                                         const tensorflow::DataType              Dtype,
                                         const tensorflow::ops::Variable::Attrs &VarAttrs,
                                         TDistrArgs &&... DistrArgs) {
            namespace tf = tensorflow;
            namespace to = tf::ops;

            const tf::Input ShapeAsInput = InputFromTensorShape(Shape);
            TDistr          Distribution = TDistr(Scope, ShapeAsInput, std::forward<TDistrArgs>(DistrArgs)...);
            to::Variable    Variable     = {Scope, Shape, Dtype, VarAttrs};
            to::Assign      Assign       = to::Assign(Scope, Variable, Distribution);
            Assigns.push_back(Assign);
            return Variable;
        }

        void operator()(tensorflow::ClientSession &Session) const;

    private:
        std::vector<tensorflow::ops::Assign> Assigns;
    };
}
