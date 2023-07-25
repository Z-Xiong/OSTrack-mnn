# Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework. (mnn)

The official implementation by pytorch:

https://github.com/botaoye/OSTrack

# 0. Download model
链接: https://pan.baidu.com/s/1nq7q5AWiQ7QYoqTmQ1TKgg?pwd=x6cj 提取码: x6cj

# 1. How to build and run it?

## modify your own CMakeList.txt
modify MNN path as yours

## build
```
$ mkdir build && cd build
$ cmake .. && make -j 
```

## run
```
$ cd build
$ ./ostrack-mnn-gcc-release [videopath(file or camera)]
```