import qbs

Product {
    type: ["tf_application"]
    Depends { name: "bazel" }
}
