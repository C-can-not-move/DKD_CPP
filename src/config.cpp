#include "config.h"

namespace dkd::cfg
{
	RunConfig defaults()
	{
		RunConfig c;
#ifdef _WIN32
		c.data_dir = "D:/DATA/CODE/C++/DKD_C++/dataset/cifar-100-binary";
		c.out_dir = "D:/DATA/CODE/C++/DKD_C++/run";
#else
		c.data_dir = "/home/chenzhijiang/userFile/dataset_all/cifar-100-binary";
#endif
		return c;
	}
}  // namespace dkd::cfg