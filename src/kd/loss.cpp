#include "kd/loss.h"

#include <algorithm>

namespace dkd::kd
{
    namespace
    {
        // mask 用 float 而不是 bool：
        // 后面要拿它做算术（t * mask、logits - 1000*mask），
        // float 直接可用，省一层类型提升。
        torch::Tensor make_gt_mask(const torch::Tensor& logits, const torch::Tensor& target)
        {
            return torch::zeros_like(logits).scatter(1, target.reshape({ -1, 1 }), 1);
        }

        torch::Tensor make_other_mask(const torch::Tensor& logits, const torch::Tensor& target)
        {
            return torch::ones_like(logits).scatter(1, target.reshape({ -1, 1 }), 0);
        }

        // 按两个 mask 求和，把 (N, C) 压成 (N, 2)：
        //  第 0 列 = 目标类的概率，第 1 列 = 其余所有类的概率和
        // 是求和降维——C维被边缘化掉了。
        torch::Tensor cat_mask(const torch::Tensor& t,
            const torch::Tensor& mask1,
            const torch::Tensor& mask2)
        {
            const auto t1 = (t * mask1).sum(1, /*keepdim=*/true);
            const auto t2 = (t * mask2).sum(1, /*keepdim=*/true);
            return torch::cat({ t1, t2 }, 1);
        }
    }

    DkdLossParts dkd_loss(
        torch::Tensor logits_student,
        torch::Tensor logits_teacher,
        torch::Tensor target,
        int64_t epoch,
        const DkdOpts& opts)
    {
        TORCH_CHECK(logits_student.dim() == 2 && logits_teacher.dim() == 2,"dkd_loss: logits 必须是 (N, C)");
        TORCH_CHECK(logits_student.sizes().vec() == logits_teacher.sizes().vec(),"dkd_loss: 师生 logits 形状不一致 ",logits_student.sizes(), " vs ", logits_teacher.sizes());
        TORCH_CHECK(target.scalar_type() == torch::kInt64,"dkd_loss: target 必须是 int64");
        TORCH_CHECK(target.size(0) == logits_student.size(0),"dkd_loss: 样本数不一致");

        const int64_t N = logits_student.size(0);
        TORCH_CHECK(N > 0, "dkd_loss: batch 不能为 0");
        TORCH_CHECK(opts.T > 0.0, "dkd_loss: T 必须 > 0");

        const double T = opts.T;
        const double T2 = T * T;

        const auto gt = make_gt_mask(logits_student, target);
        const auto other = make_other_mask(logits_student, target);

        // ---------- TCKD ----------
        const auto p_stu = torch::softmax(logits_student / T, 1);
        const auto p_tea = torch::softmax(logits_teacher / T, 1);

        const auto b_stu = cat_mask(p_stu, gt, other);      // (N, 2)
        const auto b_tea = cat_mask(p_tea, gt, other);

        // kl_div(input, target) 算的是 KL(target || exp(input))，
        // 所以第一个参数是学生的log 概率，第二个是教师的概率。
        const auto tckd = torch::nn::functional::kl_div(
            torch::log(b_stu), 
            b_tea,
            torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean))* T2;

        // ---------- NCKD ----------
        //
        // ⚠ 这里和论文 Algorithm 1 的写法【不同】，但数值严格等价。
        //
        //   论文（Algorithm 1, L411-415）—— 索引形式：
        //       pnct_stu = softmax(l_stu[1-t] / T)          → (N, C-1)
        //       pnct_tea = softmax(l_tea[1-t] / T)
        //       nckd     = kl_div(log(pnct_stu), pnct_tea)
        //
        //   实现（DKD.py）—— 偏移形式：
        //       softmax(z/T - 1000.0 * gt_mask)             → (N, C)，目标位 = 0
        //
        //   为什么换掉索引形式：
        //     ① 保持 (N, C)，不产生 (N, C-1) 的临时张量，反向也不用 scatter
        //     ② 能配 log_softmax（索引版是 log(softmax(...))，小概率处掉精度）
        //     ③ 不用做 per-sample 的布尔索引 —— C++ 里很别扭
        //
        //   为什么等价（不是近似）：
        //     目标位的概率是 exp(-1000) 在 float32 下【精确下溢成 0】，
        //     教师侧同样是 0 ⇒ kl_div 内部的 where(target > 0) 跳过该项
        //     ⇒ 求和实际只跑 C-1 项，与论文完全一致。
        //
        //   验收方式：靠 KL(p^T||p^S) = TCKD + (1 - pt_tea) * NCKD 的恒等式自检。
        //            offset 或 mask 一旦写错，那个等式会挂 —— 不必单独验这一处。
        //
        // 目标类 logit 减去 1000：
        //   【不能写 -inf】—— log_softmax 会给出 -inf，而 kl_div 里 0 * -inf = nan。
        //   减 1000 让 exp 下溢到精确的 0，数学上等价于「把目标类从归一化里剔除」。
        // ⭐不过这就是研究算法和研究程序的差异了。
        // ⭐软件工程转过来的，有些关于数值的操作还是不太敏感。
        const auto offset = 1000.0 * gt;
        const auto nckd = torch::nn::functional::kl_div(
            torch::log_softmax(logits_student / T - offset, 1),
            torch::softmax(logits_teacher / T - offset, 1),
            torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean))* T2;

        // ---------- CE + warmup ----------
		// torch::nn::functional::cross_entropy这个函数本身就会做 log_softmax，所以 logits_student 直接传进去就行。
		// 避免双重 softmax。
        const auto ce = torch::nn::functional::cross_entropy(logits_student, target)* opts.ce_weight;

        double w = 1.0;
        if (opts.warmup > 0)
        {
            w = std::min(static_cast<double>(epoch) / static_cast<double>(opts.warmup), 1.0);
        }

        DkdLossParts out;
        out.ce = ce;
        out.tckd = tckd;
        out.nckd = nckd;
        out.total = ce + w * (opts.alpha * tckd + opts.beta * nckd);
        return out;
    }

}  // namespace dkd::kd