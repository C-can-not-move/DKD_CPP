#include "DKD_C++.h"

int main()
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
	try {
		const auto cfg = dkd::cfg::defaults();
		const auto device = cfg.device_index < 0
			? torch::Device(torch::kCPU)
			: torch::Device(torch::kCUDA, cfg.device_index);

		torch::manual_seed(cfg.seed);

		const auto tr = dkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/true);
		const auto te = dkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/false);
		std::cout << "[data] train " << tr.images.size(0)<< "  test " << te.images.size(0) << "\n";

		auto model = dkd::model::h_resnet8x4(100);
		model->to(device);

		torch::optim::SGD optimizer(
			model->parameters(),
			torch::optim::SGDOptions(/*lr=*/0.05).momentum(0.9).weight_decay(5e-4));

		dkd::train::TrainConfig tcfg;
		tcfg.epochs = 240;
		tcfg.batch_size = 64;
		tcfg.log_every = 200;
		tcfg.lr_decay_stages = { 150, 180, 210 };
		tcfg.lr_decay_rate = 0.1;
		tcfg.save_path = cfg.out_dir + "/resnet8x4-baseline";
		tcfg.print();

		const auto res = dkd::train::train_classifier(
			model, 
			tr.images, 
			tr.labels, 
			te.images, 
			te.labels, 
			optimizer, 
			tcfg);
		res.print();
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;
}