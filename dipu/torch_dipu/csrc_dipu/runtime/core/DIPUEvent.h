// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>
#include <iostream>
#include <utility>

#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

#include "DIPUGuard.h"
#include "DIPUStream.h"

namespace dipu {
/*
 * DIPUEvents are movable not copyable wrappers around DIPU's events.
 * DIPUEvents are constructed lazily when first recorded.
 */
class DIPU_API DIPUEvent {
 public:
  // Constructors
  DIPUEvent() = default;

  // dipu do not support IpcEventHandle until now
  /*
  ~DIPUEvent() {
    if (isCreated()) {
      DIPUGuard guard(device_index_);
      devproxy::destroyEvent(event_);
    }
  }
  */
  ~DIPUEvent() {
    try {
      if (isCreated()) {
        DIPUGuard guard(device_index_);
        devproxy::destroyEvent(event_);
      }
    } catch (...) { /* No throw */
    }
  }

  DIPUEvent(const DIPUEvent&) = delete;
  DIPUEvent& operator=(const DIPUEvent&) = delete;

    DIPUEvent(DIPUEvent&& other) noexcept = default;
  DIPUEvent& operator=(DIPUEvent&& other) noexcept = default;
  /*
  DIPUEvent(DIPUEvent&& other) noexcept { *this = std::move(other); }

  DIPUEvent& operator=(DIPUEvent&& other) noexcept {
    if (this != &other) {
      device_index_ = other.device_index_;
      stream_id_ = other.stream_id_;
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }
  */

  explicit operator deviceEvent_t() const { return rawevent(); }

  // aclrtEvent do not support Less than operator until now

  c10::optional<at::Device> device() const {
    if (isCreated()) {
      return at::Device(dipu::DIPU_DEVICE_TYPE, device_index_);
    }
    return {};
  }

  bool isCreated() const { return event_ != nullptr; }
  c10::DeviceIndex device_index() const { return device_index_; }
  c10::StreamId stream_id() const { std::cout << "stream_id_: " << stream_id_ << std::endl; return stream_id_; }
  deviceEvent_t rawevent() const { return event_; }

  bool query() const {
    if (!isCreated()) {
      return true;
    }

    DIPUGuard guard(device_index_);
    return devproxy::getEventStatus(event_) == devapis::EventStatus::READY;
  }

  void record() { record(getCurrentDIPUStream()); }

  void recordOnce(const DIPUStream& stream) {
    if (-1 == stream_id_) {
      record(stream);
    }
  }

  void record(const DIPUStream& stream) {
    if (!isCreated()) {
      createEvent(stream.device_index());
    }
    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ",
                device_index_, " does not match recording stream's device ",
                stream.device_index(), ".");
    DIPUGuard guard(device_index_);
    devproxy::recordEvent(event_, stream.rawstream());
    stream_id_ = stream.id();
    std::cout << "record stream_id_ : " << stream_id_ << std::endl;
  }

  void wait(const DIPUStream& stream) {
    if (isCreated()) {
      DIPUGuard guard(stream.device_index());
      devproxy::streamWaitEvent(stream.rawstream(), event_);
    }
  }

  float elapsed_time(const DIPUEvent& other) const {
    TORCH_CHECK(
        isCreated() && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    devproxy::eventElapsedTime(&time_ms, event_, other.event_);
    return time_ms;
  }

  void synchronize() const {
    if (isCreated()) {
      devproxy::waitEvent(event_);
    }
  }

  // dipu do not support IpcEventHandle until now

 private:
  c10::DeviceIndex device_index_ = -1;
  c10::StreamId stream_id_ = -1;
  deviceEvent_t event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    DIPUGuard guard(device_index_);
    devproxy::createEvent(&event_);
  }
};

}  // namespace dipu
