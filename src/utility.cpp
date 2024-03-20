#include "utility.hpp"

utility::Cleanup::Cleanup(std::function<void(void *)> fn, void *ptr)
    : fn(fn), ptr(ptr) {}

utility::Cleanup::Cleanup(Nop) : fn(), ptr(nullptr) {}

utility::Cleanup::~Cleanup() {
  if (this->fn.has_value()) {
    this->fn.value()(this->ptr);
  }
}

utility::Cleanup::Cleanup(Cleanup &&other) : fn(other.fn), ptr(other.ptr) {
  other.fn = std::nullopt;
  other.ptr = nullptr;
}

utility::Cleanup &utility::Cleanup::operator=(utility::Cleanup &&other) {
  if (this->fn.has_value()) {
    this->fn.value()(this->ptr);
  }

  this->fn = other.fn;
  this->ptr = other.ptr;

  other.fn = std::nullopt;
  other.ptr = nullptr;

  return *this;
}
