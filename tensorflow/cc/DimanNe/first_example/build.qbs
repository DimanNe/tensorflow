import qbs 1.0
import TFApp

TFApp {
    name: "FirstExample"
    Group {
        name: "asdfqwer"
        files: 'main.cpp'
        fileTags: ['tf_src']
    }

    // files: [ "main.cpp" ]
}
