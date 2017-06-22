import qbs 1.0

Module {
    name: "bazel"
    cpp.compilerName: "clang++"; cpp.cxxStandardLibrary: "libstdc++";
    // cpp.debugInformation: true
    //

    // property bool useShark: false
    // property string boostHeaders: project.ide_source_tree + "/../boost/";
    // property string boostLibs: project.ide_source_tree + "/../boost/stage/lib";

    cpp.cxxLanguageVersion: "c++11"
    cpp.includePaths: [
        "/home/Void/mydevel/tensorflow",
        "/home/Void/mydevel/tensorflow/third_party/eigen3",
        "/home/Void/mydevel/tensorflow/bazel-out/local-opt/genfiles",
    ]
    //cpp.systemIncludePaths: ["/home/Void/mydevel/tensorflow", "/home/Void/mydevel/tensorflow/third_party/eigen3"]
    // cpp.includePaths: {
    //   var result = []
    //   if(useShark)
    //      result.push(boostHeaders)
    //   return result
    // }
    // cpp.cxxFlags: {
    //    var result = []
    //    if(useShark)
    //       result.push("-fopenmp=libomp", "-openmp")
    //    return result
    // }
    // cpp.libraryPaths: {
    //    var result = []
    //    if(useShark)
    //       result.push(project.ide_source_tree + "/../boost/stage/lib", "/usr/lib/x86_64-linux-gnu/")
    //    return result
    // }
    // cpp.dynamicLibraries: {
    //    var result = []
    //    if(useShark)
    //       result.push("gslcblas", "omp", "shark", "boost_serialization")
    //    return result
    // }
    // cpp.rpaths: {
    //    var result = []
    //    if(useShark)
    //       result.push(project.ide_source_tree + "/../boost/stage/lib")
    //    return result
    // }

    Depends { name: "cpp" }

    Rule {
        multiplex: true
        // outputFileTags: ["tf_application"]
        Artifact {
            fileTags: ["tf_application"]
        }
        prepare: {
            var cmd = new Command("touch", ["/home/Void/devel/qbs_rule_example"])
            cmd.description = "converting to hex:"
            cmd.highlight = "linker";
            console.log("asdfasdfasdfasdf")
           return [cmd];
        }
    }
}
