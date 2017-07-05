#include "variable_initializer.h"

namespace tfu {
    namespace tf = tensorflow;

    void TVariableInitializer::operator()(tensorflow::ClientSession &Session) const {
        for(const tensorflow::ops::Assign &Assign : Assigns) {
            std::vector<tf::Tensor> OutputsOfAssigning;
            TF_CHECK_OK(Session.Run({Assign}, &OutputsOfAssigning));
            // LOG(INFO) << OutputsOfAssigning[0].DebugString();
            // LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
            // LOG(INFO) << OutputsOfAssigning[0].flat<double>();
        }
    }
}
