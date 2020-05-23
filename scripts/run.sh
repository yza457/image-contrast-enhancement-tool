#!/bin/bash

non_zero=0

function run_cmd_with_check() {
  "$@"
  if [[ $? -ne 0 ]] 
  then
    printf "failed"
    ((non_zero++))
  fi
}


TRAVIS_BUILD_DIR=$PWD
cd ${TRAVIS_BUILD_DIR}
mkdir -p build
rm FAILED
rm PASSED
cd build; rm -rf *
cmake .. && make -j 4 VERBOSE=1
git log --pretty=format:'Name:%cn %nHash:%H%nTimestamp=%ci %n' -n 1 >> LOG

echo "****************GPU****************" >> LOG
run_cmd_with_check ./img2gpu ../in.pgm ../in.ppm >> LOG 2>&1
run_cmd_with_check diff out.pgm ../reference/out.pgm >> LOG 2>& 1
run_cmd_with_check diff out_yuv.ppm ../reference/out_yuv.ppm  >> LOG 2>& 1
run_cmd_with_check diff out_hsl.ppm ../reference/out_hsl.ppm  >> LOG 2>& 1
run_cmd_with_check diff gpu_out_grayscale.pgm ../reference/gpu_out_grayscale.pgm >> LOG 2>& 1
run_cmd_with_check diff gpu_out_yuv.ppm ../reference/gpu_out_yuv.ppm  >> LOG 2>& 1
run_cmd_with_check diff gpu_out_hsl.ppm ../reference/gpu_out_hsl.ppm  >> LOG 2>& 1

cd ../
FILENAME=`basename $PWD`

if [ "$non_zero" -eq "0"  ] 
then
  rm *.failed
  echo "" >> SUCCESS
  cp ./build/LOG ${TRAVIS_BUILD_DIR}/$FILENAME.log.success
else
  rm *.success
  echo "" >> FAILED
  cp ./build/LOG ${TRAVIS_BUILD_DIR}/$FILENAME.log.failed
fi
