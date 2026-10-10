#pragma once
#include <array>
#include <atomic>
#include <cstdint>

namespace task_priority_kinematic_control
{
// Exactly one producer and one consumer. Slots are allocated/prepared before either starts.
// Each owns a slot; the middle slot is exchanged atomically. Neither waits for the other.
template<class T>
class SnapshotBuffer
{
public:
  static_assert(std::atomic<uint32_t>::is_always_lock_free, "Snapshots require lock-free indices");
  template<class Initialize>
  void initialize(Initialize initialize)
  {
    for (auto & slot : slots_) {initialize(slot);}
    write_ = 0; read_ = 1; middle_.store(2, std::memory_order_relaxed);
  }
  T & writable() {return slots_[write_];}
  void publish()
  {
    write_ = middle_.exchange(write_ | dirty, std::memory_order_acq_rel) & index_mask;
  }
  const T * consume()
  {
    if (!(middle_.load(std::memory_order_acquire) & dirty)) {return nullptr;}
    read_ = middle_.exchange(read_, std::memory_order_acq_rel) & index_mask;
    return &slots_[read_];
  }
private:
  static constexpr uint32_t dirty = 4;
  static constexpr uint32_t index_mask = 3;
  std::array<T, 3> slots_;
  std::atomic<uint32_t> middle_{2};
  uint32_t write_ = 0;
  uint32_t read_ = 1;
};
}  // namespace task_priority_kinematic_control
