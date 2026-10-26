#include "train/trainer.h"

#include "data/transforms.h"
#include "model/checkpoint.h"

#include <algorithm>
#include <unordered_set>

namespace dkd::train
{
	TrainResult train_classifier(
		dkd::model::ResNet model,
		const torch::Tensor& train_images,
		const torch::Tensor& train_labels,
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg)
	{
		using namespace dkd::data;

		// 设备跟着模型走
		const auto device = model->parameters().front().device();

		TORCH_CHECK(train_images.dim() == 4, "train: expect (N, C, H, W)");
		TORCH_CHECK(train_images.size(0) == train_labels.size(0), "train: images / labels count mismatch");
		TORCH_CHECK(train_images.scalar_type() == torch::kUInt8, "train: expect uint8 images, got ", train_images.scalar_type());
		TORCH_CHECK(train_labels.scalar_type() == torch::kInt64, "train: expect int64 labels");
		TORCH_CHECK(cfg.batch_size > 0, "train: batch_size must be > 0");
		TORCH_CHECK(train_images.device().is_cpu(),"train_classifier: 数据必须留在 CPU —— 变换在 CPU 上做，每批只搬结果到 device");

		// 优化器必须抓的是这个模型的参数
		// 写不出 Tensor<float>，只能写 Tensor + scalar_type() 在运行时问它。
		{
			std::unordered_set<const void*> owned;
			for (const auto& g : optimizer.param_groups())
				for (const auto& t : g.params())
					owned.insert(t.data_ptr());

			for (const auto& p : model->parameters())
			{
				TORCH_CHECK(owned.count(p.data_ptr()) != 0,"train_classifier: 优化器抓的不是这个模型的参数 —— "
					                                       "要在模型构造完成之后再建优化器，参数取 model->parameters()");
			}
		}

		const auto norm = cifar100_norm();

		TrainResult res;
		const int64_t N = train_images.size(0);   

		for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch)
		{
			// 到指定轮就把 lr 乘一次 decay_rate
			// （mdistiller 用 epoch > stage，第 151 轮才生效；这里差一轮，忽略）
			if (std::find(cfg.lr_decay_stages.begin(), cfg.lr_decay_stages.end(), epoch)!= cfg.lr_decay_stages.end())
			{
				for (auto& g : optimizer.param_groups())
					g.options().set_lr(g.options().get_lr() * cfg.lr_decay_rate);
			}

			// BN 要用批统计 ⇒ train()
			model->train();

			const auto perm = torch::randperm(N, torch::kInt64);   // 洗牌就是这一行

			double  running = 0.0;
			int64_t nb = 0;

			for (int64_t start = 0; start < N; start += cfg.batch_size)
			{
				const int64_t len = std::min(cfg.batch_size, N - start);
				if (cfg.drop_last && len < cfg.batch_size) break;

				const auto idx = perm.narrow(0, start, len);        // CPU
				const auto xb = train_images.index_select(0, idx);  // CPU uint8
				const auto yb = train_labels.index_select(0, idx).to(device);

				// 变换全在 CPU 上做完，只搬这一次
				const auto xf = transform_train(xb, norm).to(device);

				// 纯净训练循环
				optimizer.zero_grad();
				const auto loss = torch::nn::functional::cross_entropy(model->forward(xf), yb);
				loss.backward();
				optimizer.step();
				// 结束

				running += loss.item<double>();
				++nb;

				if (cfg.log_every > 0 && (nb % cfg.log_every) == 0)
				{
					std::cout << "  epoch " << epoch << "  batch " << nb
						<< "  avg loss " << (running / static_cast<double>(nb)) << "\n";
				}
			}

			res.last_train_loss = (nb > 0) ? running / static_cast<double>(nb) : 0.0;

			// 验证：BN 用 running stats ⇒ eval()；不建图 ⇒ NoGradGuard
			const EvalResult eval_result = evaluate(model, val_images, val_labels, cfg);
			const double acc = eval_result.acc();

			res.final_val_acc = acc;
			if (acc > res.best_val_acc)
			{
				res.best_val_acc = acc;
				res.best_epoch = epoch;

				if (cfg.save_path)
				{
					dkd::model::save_checkpoint(model, *cfg.save_path);
					std::cout << "  → best so far, saved -> " << *cfg.save_path << ".{txt,bin}\n";
				}
			}

			std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
				<< "   lr " << optimizer.param_groups().front().options().get_lr()
				<< "   train_loss " << res.last_train_loss
				<< "   val_acc " << acc
				<< "   (" << eval_result.correct << "/" << eval_result.total << ")\n";
		}

		return res;
	}

	TrainResult train_distill(
		dkd::model::ResNet student, 
		dkd::model::ResNet teacher, 
		const torch::Tensor& train_images, 
		const torch::Tensor& train_labels, 
		const torch::Tensor& val_images, 
		const torch::Tensor& val_labels, 
		torch::optim::Optimizer& optimizer, 
		const TrainConfig& cfg, 
		const dkd::kd::DkdOpts& kd)
	{
		using namespace dkd::data;

		// 设备跟着 student 走
		const auto device = student->parameters().front().device();

		TORCH_CHECK(train_images.dim() == 4, "train_distill: expect (N, C, H, W)");
		TORCH_CHECK(train_images.size(0) == train_labels.size(0), "train_distill: images / labels count mismatch");
		TORCH_CHECK(train_images.scalar_type() == torch::kUInt8, "train_distill: expect uint8 images, got ", train_images.scalar_type());
		TORCH_CHECK(train_labels.scalar_type() == torch::kInt64, "train_distill: expect int64 labels");
		TORCH_CHECK(cfg.batch_size > 0, "train_distill: batch_size must be > 0");
		TORCH_CHECK(train_images.device().is_cpu(), "train_distill: 数据必须留在 CPU —— 变换在 CPU 上做，每批只搬结果到 device");

		// ── teacher 的三条硬约束：由调用者设，函数只当守门人 ──
		TORCH_CHECK(!teacher.is_empty(), "train_distill: teacher 为空");
		TORCH_CHECK(!teacher->is_training(),
			"train_distill: teacher 必须是 eval() 模式（BN 用 running stats）—— 由调用者设置");
		TORCH_CHECK(teacher->parameters().front().device() == device,
			"train_distill: teacher 在 ", teacher->parameters().front().device(),
			" 而 student 在 ", device);

		// 优化器必须抓的是 student 的参数
		{
			std::unordered_set<const void*> owned;
			for (const auto& g : optimizer.param_groups())
				for (const auto& t : g.params())
					owned.insert(t.data_ptr());

			for (const auto& p : student->parameters())
			{
				TORCH_CHECK(owned.count(p.data_ptr()) != 0,
					"train_distill: 优化器抓的不是 student 的参数 —— "
					"要在模型构造完成之后再建优化器，参数取 student->parameters()");
			}
		}

		const auto norm = cifar100_norm();

		TrainResult res;
		const int64_t N = train_images.size(0);

		for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch)
		{
			// 到指定轮就把 lr 乘一次 decay_rate
			if (std::find(cfg.lr_decay_stages.begin(), cfg.lr_decay_stages.end(), epoch) != cfg.lr_decay_stages.end())
			{
				for (auto& g : optimizer.param_groups())
					g.options().set_lr(g.options().get_lr() * cfg.lr_decay_rate);
			}

			// student 要用批统计 ⇒ train()；teacher 的模式由调用者保证，这里不碰
			student->train();

			const auto perm = torch::randperm(N, torch::kInt64);   // 洗牌就是这一行

			double  running = 0.0;
			double  run_ce = 0.0, run_tckd = 0.0, run_nckd = 0.0;
			int64_t nb = 0;

			for (int64_t start = 0; start < N; start += cfg.batch_size)
			{
				const int64_t len = std::min(cfg.batch_size, N - start);
				if (cfg.drop_last && len < cfg.batch_size) break;

				const auto idx = perm.narrow(0, start, len);        // CPU
				const auto xb = train_images.index_select(0, idx);  // CPU uint8
				const auto yb = train_labels.index_select(0, idx).to(device);

				// 变换全在 CPU 上做完，只搬这一次
				const auto xf = transform_train(xb, norm).to(device);

				// ⭐ teacher 前向单独包一层：构造 → 关梯度，析构 → 恢复
				//   相当于 Python 的 with torch.no_grad(): teacher(x)
				//   scope 必须收窄 —— student 的 forward / backward 在外面，否则图被丢掉。
				torch::Tensor logits_tea;
				{
					torch::NoGradGuard ng;
					logits_tea = teacher->forward(xf);
				}

				optimizer.zero_grad();

				// ce、warmup、ce_weight 全在 dkd_loss 内部算完了，
				// 这里【不要】再单独算一次 cross_entropy。
				const auto parts = dkd::kd::dkd_loss(
					student->forward(xf), logits_tea, yb, epoch, kd);

				parts.total.backward();
				optimizer.step();

				running += parts.total.item<double>();
				run_ce += parts.ce.item<double>();
				run_tckd += parts.tckd.item<double>();
				run_nckd += parts.nckd.item<double>();
				++nb;

				if (cfg.log_every > 0 && (nb % cfg.log_every) == 0)
				{
					std::cout << "  epoch " << epoch << "  batch " << nb
						<< "  avg total " << (running / static_cast<double>(nb)) << "\n";
				}
			}

			const double inv_nb = (nb > 0) ? 1.0 / static_cast<double>(nb) : 0.0;
			res.last_train_loss = running * inv_nb;

			// warmup 系数：第 1 轮 = 1/warmup，第 warmup 轮才到 1.0
			// ⚠ kd.warmup 是 int64_t —— 不做 double 转换会走整数除法，全程得 0
			const double w = (kd.warmup > 0)
				? std::min(static_cast<double>(epoch) / static_cast<double>(kd.warmup), 1.0)
				: 1.0;

			// 验证：BN 用 running stats ⇒ eval()；不建图 ⇒ NoGradGuard
			const EvalResult eval_result = evaluate(student, val_images, val_labels, cfg);
			const double acc = eval_result.acc();

			res.final_val_acc = acc;
			if (acc > res.best_val_acc)
			{
				res.best_val_acc = acc;
				res.best_epoch = epoch;

				if (cfg.save_path)
				{
					dkd::model::save_checkpoint(student, *cfg.save_path);
					std::cout << "  → best so far, saved -> " << *cfg.save_path << ".{txt,bin}\n";
				}
			}

			std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
				<< "   lr " << optimizer.param_groups().front().options().get_lr()
				<< "   ce " << run_ce * inv_nb
				<< "   tckd " << run_tckd * inv_nb
				<< "   nckd " << run_nckd * inv_nb
				<< "   w " << w
				<< "   total " << res.last_train_loss
				<< "   acc " << acc
				<< "   (" << eval_result.correct << "/" << eval_result.total << ")\n";
		}

		return res;
	}

	EvalResult evaluate(
		dkd::model::ResNet model,
		const torch::Tensor& images,
		const torch::Tensor& labels,
		const TrainConfig& cfg)
	{
		TORCH_CHECK(cfg.batch_size > 0, "evaluate: batch_size must be > 0");
		TORCH_CHECK(images.size(0) == labels.size(0), "evaluate: images / labels count mismatch");
		TORCH_CHECK(images.device().is_cpu(), "evaluate: 数据必须留在 CPU");

		const auto device = model->parameters().front().device();
		const auto norm = dkd::data::cifar100_norm();

		torch::NoGradGuard no_grad;
		const bool was_training = model->is_training();   // is_training() 是 const
		model->eval();                                    // 会改变共享 impl 的模式

		int64_t correct = 0, total = 0;
		for (int64_t s = 0; s < images.size(0); s += cfg.batch_size)
		{
			const int64_t len = std::min(cfg.batch_size, images.size(0) - s);
			const auto xb = images.narrow(0, s, len);
			const auto yb = labels.narrow(0, s, len).to(device);
			const auto xf = dkd::data::transform_val(xb, norm).to(device);

			const auto pred = model->forward(xf).argmax(1);
			correct += pred.eq(yb).sum().item<int64_t>();
			total += len;
		}

		if (was_training) model->train();                 // 恢复调用者进来时的状态

		EvalResult res;
		res.correct = correct;
		res.total = total;
		return res;
	}

}  // namespace dkd::train