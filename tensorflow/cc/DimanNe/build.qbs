import qbs 1.0
Project {
   name: "MyTFProjects"
   qbsSearchPaths: ["qbs_templates"]

   references: [
      "first_example/build.qbs"
   ]
}
