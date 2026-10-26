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
		std::cout << "[data] train " << tr.images.size(0)
			<< "  test " << te.images.size(0) << "\n";

		// ── teacher：作者发布的 resnet32x4（CIFAR-100 79.42%）────────
		auto teacher = dkd::model::h_resnet32x4(100);
		dkd::model::load_checkpoint(teacher, "D:/DATA/CODE/C++/DKD_C++/tool/resnet32x4_100");
		teacher->to(device);
		teacher->eval();          // 必须由调用者设，train_distill 里会校验

		// ── student：从零训，【不】加载基线权重 ──────────────────────
		auto student = dkd::model::h_resnet8x4(100);
		student->to(device);

		torch::optim::SGD optimizer(
			student->parameters(),                // ← 必须是 student 的参数
			torch::optim::SGDOptions(/*lr=*/0.05).momentum(0.9).weight_decay(5e-4));

		dkd::train::TrainConfig tcfg;
		tcfg.epochs = 240;
		tcfg.batch_size = 64;
		tcfg.log_every = 200;
		tcfg.lr_decay_stages = { 150, 180, 210 };
		tcfg.lr_decay_rate = 0.1;
		tcfg.save_path = cfg.out_dir + "/resnet8x4-dkd";
		tcfg.print();

		dkd::kd::DkdOpts kd;      // α=1  β=8  T=4  ce_weight=1  warmup=20
		std::cout << "[dkd-cfg] alpha " << kd.alpha
			<< "  beta " << kd.beta
			<< "  T " << kd.T
			<< "  ce_weight " << kd.ce_weight
			<< "  warmup " << kd.warmup << "\n";

		const auto res = dkd::train::train_distill(
			student, teacher,
			tr.images, tr.labels, te.images, te.labels,
			optimizer, tcfg, kd);
		res.print();
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}
	return 0;
}