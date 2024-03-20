#ifndef DITHERING_UTILITY_HPP
#define DITHERING_UTILITY_HPP

#include <cmath>
#include <functional>
#include <optional>
#include <utility>

namespace utility {
inline int twoToOne(int x, int y, int width, int height) {
  while (x < 0) {
    x += width;
  }
  while (y < 0) {
    y += height;
  }
  x = x % width;
  y = y % height;
  return x + y * width;
}

inline std::pair<int, int> oneToTwo(int i, int width) {
  return {i % width, i / width};
}

inline float dist(int a, int b, int width) {
  auto axy = utility::oneToTwo(a, width);
  auto bxy = utility::oneToTwo(b, width);
  float dx = axy.first - bxy.first;
  float dy = axy.second - bxy.second;
  return std::sqrt(dx * dx + dy * dy);
}

class Cleanup {
 public:
  struct Nop {};

  Cleanup(std::function<void(void *)> fn, void *ptr);
  Cleanup(Nop nop);
  ~Cleanup();

  // allow move
  Cleanup(Cleanup &&);
  Cleanup &operator=(Cleanup &&);

  // deny copy
  Cleanup(const Cleanup &) = delete;
  Cleanup &operator=(const Cleanup &) = delete;

 private:
  std::optional<std::function<void(void *)>> fn;
  void *ptr;
};
}  // namespace utility

#endif
