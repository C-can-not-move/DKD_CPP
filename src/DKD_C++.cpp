#include "DKD_C++.h"

// ==================== dkd_loss 的自检 ====================
static void test_dkd_loss()
{
    using namespace dkd::kd;

    torch::manual_seed(0);
    const int64_t C = 10;
    const double  T = 4.0;
    const DkdOpts opts;            // alpha=1  beta=8  T=4  warmup=20

    std::cout << std::fixed << std::setprecision(8);

    // ---------- ① 恒等式：KL = TCKD + pt_tea * NCKD（用 N=1，避免加权被平均掉） ----------
    {
        auto zs = torch::randn({ 1, C }) * 2.0;
        auto zt = torch::randn({ 1, C }) * 2.0;
        auto y = torch::randint(0, C, { 1 }, torch::kInt64);

        const auto p = dkd_loss(zs, zt, y, /*epoch=*/20);   // w = 1.0

        const auto logp_s = torch::log_softmax(zs / T, 1);
        const auto logp_t = torch::log_softmax(zt / T, 1);
        const auto kl = (torch::exp(logp_t) * (logp_t - logp_s)).sum().item<double>() * T * T;

        const auto pt_tea = torch::exp(logp_t).gather(1, y.reshape({ -1, 1 })).item<double>();
        const auto recon = p.tckd.item<double>() + (1.0-pt_tea) * p.nckd.item<double>();

        std::cout << "[1] 恒等式  KL=" << kl << "   TCKD+pt*NCKD=" << recon
            << "   diff=" << std::abs(kl - recon) << "\n";
    }

    // ---------- ② batchmean：一次 N=8  ==  8 次 N=1 的平均 ----------
    auto zs8 = torch::randn({ 8, C }) * 2.0;
    auto zt8 = torch::randn({ 8, C }) * 2.0;
    auto y8 = torch::randint(0, C, { 8 }, torch::kInt64);

    const auto batch = dkd_loss(zs8, zt8, y8, /*epoch=*/20);
    {
        double s_tckd = 0, s_nckd = 0, s_ce = 0;
        for (int64_t i = 0; i < 8; ++i)
        {
            const auto one = dkd_loss(zs8[i].unsqueeze(0), zt8[i].unsqueeze(0),
                y8[i].unsqueeze(0), 20);
            s_tckd += one.tckd.item<double>();
            s_nckd += one.nckd.item<double>();
            s_ce += one.ce.item<double>();
        }
        std::cout << "[2] batchmean  ce " << batch.ce.item<double>()
            << " vs " << s_ce / 8 << "\n";
        std::cout << "              tckd " << batch.tckd.item<double>()
            << " vs " << s_tckd / 8 << "\n";
        std::cout << "              nckd " << batch.nckd.item<double>()
            << " vs " << s_nckd / 8 << "\n";
    }

    // ---------- ③ CE 独立核对：logsumexp(z) - z_y，且【不带温度】 ----------
    {
        const auto ce_each = torch::logsumexp(zs8, 1)
            - zs8.gather(1, y8.reshape({ -1, 1 })).squeeze(1);
        std::cout << "[3] CE     " << batch.ce.item<double>()
            << " vs " << ce_each.mean().item<double>() << "\n";
    }

    // ---------- ④ total 的组装：total == ce + 1.0*(alpha*tckd + beta*nckd) ----------
    {
        const auto expect = batch.ce
            + (opts.alpha * batch.tckd + opts.beta * batch.nckd);
        std::cout << "[4] total  " << batch.total.item<double>()
            << "   diff=" << (batch.total - expect).abs().item<double>() << "\n";
    }

    // ---------- ⑤ warmup：从 total 反解 w ----------
    {
        const int64_t eps[] = { 1, 5, 10, 20, 40 };
        std::cout << "[5] warmup\n";
        for (int64_t ep : eps)
        {
            const auto p = dkd_loss(zs8, zt8, y8, ep);
            const double denom = (opts.alpha * p.tckd + opts.beta * p.nckd).item<double>();
            const double w = (p.total - p.ce).item<double>() / denom;
            std::cout << "      epoch " << std::setw(2) << ep
                << "  w = " << w << "\n";
        }
    }
}

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
	try
	{
        test_dkd_loss();
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;

}