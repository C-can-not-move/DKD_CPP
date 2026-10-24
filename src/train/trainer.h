// src/train/trainer.h
#pragma once

#include "model/resnet.h"

#include <torch/torch.h>
#include <torch/optim.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

namespace dkd::train
{
	struct TrainConfig
	{
		int64_t epochs = 20;           // 官方是 240
		int64_t batch_size = 64;       // mdistiller 的 CIFAR-100 配置
		bool    drop_last = false;
		int64_t log_every = 50;
		std::optional<std::string> save_path;

		// 没有 out_size / resize_to —— CIFAR 的 32x32 是固定的，
		// 网络（AvgPool2d(8)）也只吃这个尺寸。

		void print() const
		{
			std::cout << "[train-cfg] epochs     = " << epochs << '\n'
				<< "[train-cfg] batch_size = " << batch_size << '\n'
				<< "[train-cfg] drop_last  = " << (drop_last ? "true" : "false") << '\n'
				<< "[train-cfg] log_every  = " << log_every << '\n'
				<< "[train-cfg] save_path  = " << (save_path ? save_path->c_str() : "(none)") << '\n';
		}
	};

	struct EvalResult
	{
		int64_t correct = 0;
		int64_t total = 0;

		double acc() const
		{
			return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
		}
	};

	struct TrainResult
	{
		double  best_val_acc = 0.0;
		int64_t best_epoch = 0;
		double  final_val_acc = 0.0;
		double  last_train_loss = 0.0;

		void print() const
		{
			std::cout << "best_val_acc = " << best_val_acc
				<< "  (epoch " << best_epoch << ")"
				<< "   final_val_acc = " << final_val_acc
				<< "   last_train_loss = " << last_train_loss << '\n';
		}
	};

	// 从头训一个分类器。
	// 数据是整批张量，函数内部自己洗牌切片，不用 DataLoader。
	TrainResult train_classifier(
		dkd::model::ResNet model,
		const torch::Tensor& train_images,   // uint8 (N, 3, 32, 32)，CPU
		const torch::Tensor& train_labels,   // int64 (N)，CPU
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg);

	// 评估准确率，没有训练循环。
	EvalResult evaluate(
		dkd::model::ResNet model,
		const torch::Tensor& images,
		const torch::Tensor& labels,
		const TrainConfig& cfg);

}  // namespace dkd::train