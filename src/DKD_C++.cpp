#include "DKD_C++.h"

static const std::string kDataDir ="D:/DATA/CODE/C++/DKD_C++/dataset/cifar-100-binary";

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try
    {
        const auto tr = dkd::data::load_cifar100_bin(kDataDir, /*train=*/true);
        const auto te = dkd::data::load_cifar100_bin(kDataDir, /*train=*/false);

        std::cout << "train images " << tr.images.sizes() << "  " << tr.images.scalar_type() << "\n";
        std::cout << "train labels " << tr.labels.sizes() << "  " << tr.labels.scalar_type() << "\n";
        std::cout << "test  images " << te.images.sizes() << "\n";
        std::cout << "test  labels " << te.labels.sizes() << "\n";

        std::cout << "label  min " << tr.labels.min().item<int64_t>()
            << "  max " << tr.labels.max().item<int64_t>() << "\n";
        std::cout << "pixel  min " << tr.images.min().item<int64_t>()
            << "  max " << tr.images.max().item<int64_t>() << "\n";

        // RGB 平面排布，前 8 个是 R 平面的前 8 个
        std::cout << "第一张前 8 个像素 "
            << tr.images[0].flatten().slice(0, 0, 8) << "\n";

        // 每类计数：训练集 100 类应全部等于 500
        auto counts = torch::zeros({ 100 }, torch::TensorOptions().dtype(torch::kInt64));
        counts.scatter_add_(0, tr.labels, torch::ones_like(tr.labels));
        std::cout << "每类计数 min " << counts.min().item<int64_t>()
            << "  max " << counts.max().item<int64_t>() << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}