#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

namespace dipu {
// just an example
// not implemented now
class DROPLETGeneratorImpl : public dipu::DIPUGeneratorImpl {
public:
  DROPLETGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }

  void init_state() const override {
  }

  void set_state(const c10::TensorImpl& state) override {
  }

  void update_state() const override {
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<DROPLETGeneratorImpl>(device_index);
}

}