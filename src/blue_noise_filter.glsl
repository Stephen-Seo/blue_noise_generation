#version 450

struct FloatAndIndex {
  float value;
  int pbp;
  int idx;
};

int twoToOne(int x, int y, int width, int height) {
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

layout(binding = 0) readonly buffer PreComputed { float precomputed[]; };

layout(binding = 1) buffer FilterInOut { FloatAndIndex filter_in_out[]; };

layout(binding = 2) readonly buffer Other {
  int width;
  int height;
  int filter_size;
};

layout(local_size_x = 256) in;

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= width * height) {
    return;
  }

  // filter_in_out[index].idx = int(index);

  int x = int(index % width);
  int y = int(index / width);

  filter_in_out[index].value = 0.0F;
  for (int q = 0; q < filter_size; ++q) {
    int q_prime = height - filter_size / 2 + y + q;
    for (int p = 0; p < filter_size; ++p) {
      int p_prime = width - filter_size / 2 + x + p;
      if (filter_in_out[twoToOne(p_prime, q_prime, width, height)].pbp != 0) {
        filter_in_out[index].value +=
            precomputed[twoToOne(p, q, filter_size, filter_size)];
      }
    }
  }
}
