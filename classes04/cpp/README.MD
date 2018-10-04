## 编译问题
1.  /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/stdlib.h:94:15: fatal error: 'stdlib.h' file not found
#include_next <stdlib.h>
              ^~~~~~~~~~
1 error generated.
make[2]: *** [dlib_build/CMakeFiles/dlib.dir/gui_widgets/fonts.cpp.o] Error 1
make[1]: *** [dlib_build/CMakeFiles/dlib.dir/all] Error 2
make: *** [all] Error 2
https://github.com/frida/frida/issues/338