var File = require("qbs.File");
var FileInfo = require("qbs.FileInfo");
var TextFile = require("qbs.TextFile");

function getTensorFlowRoot(srcDir) {
    for(var path = srcDir; path != "/"; path = FileInfo.path(path)) {
        if(File.exists(path + "/.git"))
            return path
    }
    throw "Can not find tensorflow root (.git folder) from " + srcDir
}

function getRelativePath(fullPath) {
    var tensorFlowRootPath = getTensorFlowRoot(fullPath);
    var result = FileInfo.relativePath(tensorFlowRootPath, fullPath);
    return result;
}
function getBazelProjectName(fullPath) {
    var relativePath = getRelativePath(fullPath)
    var result = relativePath + ":" + FileInfo.fileName(relativePath);
    return result;
}

// function generateBUILDFile(tfSrcs, tfDeps, projectName, directory) {
//     var file = new TextFile(directory + "/BUILD", TextFile.WriteOnly);
//     file.write("cc_binary(\n");
//     file.write("   name = \"" + projectName + "\",\n");
//
//
//     file.write("   srcs = [ ");
//     for(var i = 0; i < tfSrcs.length; ++i) {
//         if(i != 0)
//             file.write(", ");
//         file.write("\"" + tfSrcs[i].fileName + "\"");
//     }
//     file.write(" ],\n");
//
//
//     file.write("   deps = [ ");
//     for(var i = 0; i < tfDeps.length; ++i) {
//         if(i != 0)
//             file.write(",\n");
//         file.write("\"" + tfDeps[i].fileName + "\"");
//     }
//     file.write(" ],\n");
//
//
//     file.write(")\n");
//     file.close();
// }
