// DKD_C++.h: 标准系统包含文件的包含文件

// window 控制台显示 utf-8
#ifdef _WIN32
#include <windows.h>
#endif

#pragma once

#include <iostream>
#include <string>

#include <torch/torch.h>

#include "data/cifar100.h"
#include "model/resnet.h"