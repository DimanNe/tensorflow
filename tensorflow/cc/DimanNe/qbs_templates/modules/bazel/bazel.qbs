import qbs 1.0
import "bazel_build_helpers.js" as bbh

Module {
    name: "bazel"

    Depends { name: "cpp" }
    cpp.compilerName: "clang++"; cpp.cxxStandardLibrary: "libstdc++";
    cpp.cxxLanguageVersion: "c++11"
    //cpp.systemIncludePaths: ["/home/Void/mydevel/tensorflow", "/home/Void/mydevel/tensorflow/third_party/eigen3"]
    cpp.includePaths: [
        "/home/Void/mydevel/tensorflow",
        "/home/Void/mydevel/tensorflow/third_party/eigen3",
        "/home/Void/mydevel/tensorflow/bazel-out/local-opt/genfiles",
    ]

    // =============================================================================================================

    Rule {
        alwaysRun: true
        multiplex: true
        inputs: ["tf_src" ]
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
            var cmd = new Command("bazel", [
                                      "run",
                                      "-c",
                                      "opt",
                                      "--copt=-mavx",
                                      "--copt=-mavx2",
                                      "--copt=-mfma",
                                      "--copt=-mfpmath=both",
                                      "--copt=-msse4.2",
                                      bazelProjectName
                                  ])
            cmd.workingDirectory = tensorFlowRoot;
            cmd.description = "Building " + bazelProjectName; //product.sourceDirectory;
            cmd.highlight = "compiler";
           return [cmd];
        }
    }
}
