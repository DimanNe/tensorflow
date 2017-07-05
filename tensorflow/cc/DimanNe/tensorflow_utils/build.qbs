import qbs 1.0

Product {
    name: "TensorFlowUtils"
    type: ["tf_sources"]
    Depends { name: "bazel" }

    files: [
        "variable_initializer.h",
        "variable_initializer.cpp",
        "shape_helpers.cpp",
        "shape_helpers.h",
        "context.h",
        "context.cpp"
    ]
}
