cc_library(
    name = "tensorflow_utils",
    srcs = [
       "variable_initializer.cpp",
       "variable_initializer.h",
       "shape_helpers.cpp",
       "shape_helpers.h",
       "context.h",
       "context.cpp"
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
    visibility = ["//visibility:public"]
)
