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

		auto teacher = dkd::model::h_resnet32x4(100);
		dkd::model::load_checkpoint(teacher, "D:/DATA/CODE/C++/DKD_C++/tool/resnet32x4_100");
		teacher->to(device);

		const auto te = dkd::data::load_cifar100_bin(cfg.data_dir, /*train=*/false);
		std::cout << "[data] test " << te.images.size(0) << " 张\n";

		dkd::train::TrainConfig tcfg;
		tcfg.batch_size = 100;

		const auto r = dkd::train::evaluate(teacher, te.images, te.labels, tcfg);
		std::cout << "[eval] correct " << r.correct << " / " << r.total
			<< "   acc " << r.acc() * 100.0 << " %\n";
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}

	if (0)
	{
		try{
			const auto cfg_all = dkd::cfg::defaults();
			const auto device = cfg_all.device_index < 0
				? torch::Device(torch::kCPU)
				: torch::Device(torch::kCUDA, cfg_all.device_index);

			torch::manual_seed(cfg_all.seed);

			auto tr = dkd::data::load_cifar100_bin(cfg_all.data_dir, /*train=*/true);
			auto te = dkd::data::load_cifar100_bin(cfg_all.data_dir, /*train=*/false);

			auto model = dkd::model::h_resnet8x4(100);
			model->to(device);

			torch::optim::SGD optimizer(
				model->parameters(),
				torch::optim::SGDOptions(/*lr=*/0.05).momentum(0.9).weight_decay(5e-4));

			dkd::train::TrainConfig cfg;
			cfg.epochs = 240;
			cfg.batch_size = 64;
			cfg.log_every = 200;                       // 别刷屏，782 批/epoch
			cfg.lr_decay_stages = { 150, 180, 210 };
			cfg.lr_decay_rate = 0.1;
			cfg.print();

			const auto res = dkd::train::train_classifier(
				model,
				tr.images,
				tr.labels,
				te.images,
				te.labels,
				optimizer,
				cfg);

			res.print();
		}
		catch (const std::exception& e){
			std::cerr << "ERROR: " << e.what() << "\n";
			return 1;
		}
	}

	return 0;
}