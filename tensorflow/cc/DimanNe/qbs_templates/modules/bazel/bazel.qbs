import qbs 1.0
import "bazel_build_helpers.js" as bbh

Module {
    name: "bazel"

    Depends { name: "cpp" }
    cpp.compilerName: "clang++"; cpp.cxxStandardLibrary: "libstdc++";
    cpp.cxxLanguageVersion: "c++11"
    //cpp.systemIncludePaths: ["/home/Void/mydevel/tensorflow", "/home/Void/mydevel/tensorflow/third_party/eigen3"]

    property string tensorFlowRootPath: bbh.getTensorFlowRoot(product.sourceDirectory)
    cpp.includePaths: [
        tensorFlowRootPath,

        tensorFlowRootPath + "/bazel-tensorflow/external/protobuf/src",
        // tensorFlowRootPath + "/third_party/eigen3",
        // tensorFlowRootPath + "/bazel-out/local-opt/genfiles",
        // tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/eigen_archive",

        tensorFlowRootPath + "/bazel-tensorflow/external/protobuf",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/bazel_tools",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/boringssl",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/com_googlesource_code_re2",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/curl",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/eigen_archive",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/farmhash_archive",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/fft2d",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/gemmlowp",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/gif_archive",

        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/highwayhash",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/jemalloc",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/jpeg",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/jsoncpp_git",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/local_config_cuda",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/local_config_sycl",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/png_archive",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/protobuf",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/snappy",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/zlib_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/bazel_tools",
        tensorFlowRootPath + "/bazel-tensorflow/external/boringssl",
        tensorFlowRootPath + "/bazel-tensorflow/external/com_googlesource_code_re2",

        tensorFlowRootPath + "/bazel-tensorflow/external/curl",
        tensorFlowRootPath + "/bazel-tensorflow/external/eigen_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/farmhash_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/fft2d",
        tensorFlowRootPath + "/bazel-tensorflow/external/gemmlowp",
        tensorFlowRootPath + "/bazel-tensorflow/external/gif_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/highwayhash",
        tensorFlowRootPath + "/bazel-tensorflow/external/jemalloc",
        tensorFlowRootPath + "/bazel-tensorflow/external/jpeg",

        tensorFlowRootPath + "/bazel-tensorflow/external/jsoncpp_git",
        tensorFlowRootPath + "/bazel-tensorflow/external/local_config_cuda",
        tensorFlowRootPath + "/bazel-tensorflow/external/local_config_sycl",
        tensorFlowRootPath + "/bazel-tensorflow/external/png_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/snappy",
        tensorFlowRootPath + "/bazel-tensorflow/external/zlib_archive"
    ]

    systemIncludePaths: [
        tensorFlowRootPath + "/bazel-tensorflow/external/protobuf/src",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/boringssl/src/include",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/curl/include",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/eigen_archive",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/farmhash_archive/src",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/gif_archive/lib",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/jemalloc/include",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/jsoncpp_git/include",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/local_config_cuda/cuda",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/local_config_cuda/cuda/include",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/png_archive",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/protobuf/src",
        tensorFlowRootPath + "/bazel-out/local-opt/genfiles/external/zlib_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/bazel_tools/tools/cpp/gcc3",
        tensorFlowRootPath + "/bazel-tensorflow/external/boringssl/src/include",
        tensorFlowRootPath + "/bazel-tensorflow/external/curl/include",
        tensorFlowRootPath + "/bazel-tensorflow/external/eigen_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/farmhash_archive/src",
        tensorFlowRootPath + "/bazel-tensorflow/external/gif_archive/lib",
        tensorFlowRootPath + "/bazel-tensorflow/external/jemalloc/include",
        tensorFlowRootPath + "/bazel-tensorflow/external/jsoncpp_git/include",
        tensorFlowRootPath + "/bazel-tensorflow/external/local_config_cuda/cuda",
        tensorFlowRootPath + "/bazel-tensorflow/external/local_config_cuda/cuda/include",
        tensorFlowRootPath + "/bazel-tensorflow/external/png_archive",
        tensorFlowRootPath + "/bazel-tensorflow/external/zlib_archive"
    ]



    defines: [
        "EIGEN_MPL2_ONLY",
        "TENSORFLOW_USE_JEMALLOC"
    ]


    // =============================================================================================================

    Rule {
        alwaysRun: true
        multiplex: true
        // inputs: ["tf_src" ]
        Artifact {
            fileTags: ["tf_application"]
        }

        prepare: {
            // throw "getRelativePath: " + bbh.getRelativePath(product.sourceDirectory)
            //throw bbh.getBazelProjectName(product.sourceDirectory)
            // throw inputs["tf_src"][0]
            // bbh.generateBUILDFile(inputs["tf_src"], inputs["tf_deps"], "qwerProj", product.sourceDirectory);




            var tensorFlowRoot = bbh.getTensorFlowRoot(product.sourceDirectory)
            var bazelProjectName = bbh.getBazelProjectName(product.sourceDirectory);
            // bazel run -c opt --copt=-mavx --copt=-msse4.2
            var cmd = new Command("bazel", [
                                       "build",
                                       "-c",
                                       "dbg", // "opt",
                                       "--copt=-mavx",
                                      "--copt=-g",
                                       // "--copt=-mavx2",
                                       //"--copt=-mfma",
                                       //"--copt=-mfpmath=both",
                                       "--copt=-msse4.2",
                                       "-s",
                                       bazelProjectName
                                   ]);
            cmd.workingDirectory = tensorFlowRoot;
            cmd.description = "Building " + bazelProjectName; //product.sourceDirectory;
            cmd.highlight = "compiler";
            return [cmd];
        }
    }
}
