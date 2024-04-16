// Copyright (c) 2024, DeepLink.

#include "csrc_dipu/runtime/distributed/distributedUtil.h"

#include "csrc_dipu/runtime/distributed/UtilInstance.h"

namespace dipu {

class AscendDistributedUtil : public distributedUtil {
 public:
  void allreducePreFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                      std::vector<at::Tensor>& inputs,
                      std::vector<at::Tensor>& outputs) override {
    if (inputs[0].scalar_type() == at::kBool ||
        inputs[0].scalar_type() == at::kByte) {
      DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
      outputs[0] = inputs[0].to(at::kInt);
      std::cout << "come into process pre" << std::endl;
    }
    std::cout << "come into AscendDistributedUtil allreducePreFn impl."
              << std::endl;
  }

  void allreducePostFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                       std::vector<at::Tensor>& inputs,
                       std::vector<at::Tensor>& outputs) override {
    if (inputs[0].scalar_type() != outputs[0].scalar_type()) {
      DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
      outputs[0].copy_(inputs[0]);
      std::cout << "come into process post" << std::endl;
    }
    std::cout << "come into AscendDistributedUtil allreducePostFn impl."
              << std::endl;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static AscendDistributedUtil util;

const static int32_t ascend_init = []() {
  UtilInstance::getInstance().setVendorImpl(&util);
  return 1;
}();

}  // namespace dipu
