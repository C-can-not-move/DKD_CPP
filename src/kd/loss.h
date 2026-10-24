// src/kd/loss.h
#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace dkd::kd
{
    struct DkdOpts
    {
        double  alpha = 1.0;
        double  beta = 8.0;
        double  T = 4.0;
        double  ce_weight = 1.0;  // 恒为 1.0 —— 这是 DKD 相对经典 KD 的关键差别（KD 用 0.1）
        int64_t warmup = 20;      // 先让 CE 说话，等学生有了基本判断，再逐渐把师傅的声音放大
    };

    // 每一项都是标量。tckd / nckd 已经乘过 T²，但【没有】乘 α / β。
    struct DkdLossParts
    {
        torch::Tensor ce;
        torch::Tensor tckd;
        torch::Tensor nckd;
        torch::Tensor total;
    };

    // epoch从1开始—— mdistiller 的 trainer.py 从 1 开始数。
    // 所以第 1 轮的 warmup 系数是 1/20 = 0.05，第 20 轮才到 1.0。
    // DKD 这里是「学生用 log_softmax、教师用 softmax」
    DkdLossParts dkd_loss(
        torch::Tensor logits_student,   // (N, C) logits，未经 softmax
        torch::Tensor logits_teacher,   // (N, C) logits，未经 softmax
        torch::Tensor target,           // (N,)   int64
        int64_t epoch,
        const DkdOpts& opts = {});

}  // namespace dkd::kd