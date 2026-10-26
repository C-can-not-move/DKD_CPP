// DKD_C++.h: 标准系统包含文件的包含文件

// window 控制台显示 utf-8
#ifdef _WIN32
#include <windows.h>
#endif

#pragma once

#include <iostream>
#include <iomanip>
#include <string>

#include <fstream>
#include <set>
#include <cstdio>

#include <torch/torch.h>

#include "config.h"
#include "data/cifar100.h"
#include "data/transforms.h"
#include "model/resnet.h"
#include "model/checkpoint.h"
#include "kd/loss.h"
#include "train/trainer.h"