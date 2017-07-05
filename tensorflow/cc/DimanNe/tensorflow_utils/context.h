#pragma once

#include "variable_initializer.h"

#include "tensorflow/cc/framework/scope.h"

namespace tfu {
    struct TContext {
        TVariableInitializer InitV;
    };
}
