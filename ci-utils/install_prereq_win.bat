mkdir local_install
mkdir local_install\include

curl -L https://github.com/PolusAI/filepattern/archive/refs/tags/v2.0.4.zip -o v2.0.4.zip 
tar -xvf v2.0.4.zip
pushd filepattern-2.0.4
mkdir build
pushd build
cmake -Dfilepattern_SHARED_LIB=ON -DCMAKE_PREFIX_PATH=../../local_install -DCMAKE_INSTALL_PREFIX=../../local_install ../src/filepattern/cpp
cmake --build . --config Release --target install --parallel 4
popd
popd

curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip -o v2.11.1.zip
tar -xvf v2.11.1.zip
pushd pybind11-2.11.1
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/  -DPYBIND11_TEST=OFF ..
cmake --build . --config Release --target install  
popd
popd

if errorlevel 1 exit 1

if "%ON_GITHUB%"=="TRUE" xcopy /E /I /y local_install\bin %TEMP%\argolid\bin