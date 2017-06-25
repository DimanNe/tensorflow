import qbs 1.0
Project {
   name: "MyTFProjects"
   qbsSearchPaths: ["qbs_templates"]
   references: [
      "tensorflow-sources/build.qbs",
      "first_example/build.qbs"
   ]
}
