#include "DKD_C++.h"

int main()
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
    auto a = dkd::model::h_resnet8x4(100);         // 随机初始化
    dkd::model::save_checkpoint(a, "run/roundtrip");

    auto b = dkd::model::h_resnet8x4(100);         // 另一份随机初始化，值必然不同
    dkd::model::load_checkpoint(b, "run/roundtrip");

    std::unordered_map<std::string, torch::Tensor> tb;
    for (auto& p : b->named_parameters()) tb.emplace(p.key(), p.value());
    for (auto& x : b->named_buffers())    tb.emplace(x.key(), x.value());

    int64_t bad = 0;
    auto cmp = [&](const std::string& k, const torch::Tensor& v)
        {
            if (!v.equal(tb.at(k))) { ++bad; std::cout << "  [roundtrip] 不一致 " << k << "\n"; }
        };
    for (auto& p : a->named_parameters()) cmp(p.key(), p.value());
    for (auto& x : a->named_buffers())    cmp(x.key(), x.value());

    std::cout << "[roundtrip] " << (bad == 0 ? "全等 ✓" : "不一致 ✗")
        << "   (" << tb.size() << " 个张量)\n";

    return 0;
}