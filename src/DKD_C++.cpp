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
		auto s = dkd::model::h_resnet8x4(100);
		auto t = dkd::model::h_resnet32x4(100);

		const auto count = [](const dkd::model::ResNet& net)
			{
				int64_t n = 0;
				for (const auto& p : net->parameters()) n += p.numel();
				return n;
			};

		std::cout << "resnet8x4  参数量 " << count(s) << "  （期望 1,233,540）\n";
		std::cout << "resnet32x4 参数量 " << count(t) << "  （期望 7,433,860）\n"; 

		auto x = torch::zeros({ 2, 3, 32, 32 });
		auto y = s->forward(x);
		std::cout << "logits " << y.sizes() << "  " << y.scalar_type() << "\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;

}