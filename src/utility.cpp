#include "utility.hpp"

utility::Cleanup::Cleanup(std::function<void(void *)> fn, void *ptr)
    : fn(fn), ptr(ptr) {}

utility::Cleanup::~Cleanup() { this->fn(this->ptr); }
