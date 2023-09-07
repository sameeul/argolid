mkdir local_install
mkdir local_install\include

curl -L https://github.com/PolusAI/filepattern/archive/refs/heads/master.zip -o filepattern.zip
tar -xvf filepattern.zip
pushd filepattern-master
mkdir build
pushd build
cmake -Dfilepattern_SHARED_LIB=ON -DCMAKE_PREFIX_PATH=../../local_install -DCMAKE_INSTALL_PREFIX=../../local_install ..
cmake --build . --config Release --target install --parallel 4
popd
popd


if errorlevel 1 exit 1

if "%ON_GITHUB%"=="TRUE" xcopy /E /I /y local_install\bin %TEMP%\omepyrgen\bin