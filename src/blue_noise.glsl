#version 450

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

layout(std140, binding = 0) readonly buffer PreComputed {
  float precomputed[];
};

layout(std140, binding = 1) writeonly buffer FilterOut { float filter_out[]; };

layout(std140, binding = 2) readonly buffer PBP { int pbp[]; };

layout(std140, binding = 3) readonly buffer Other {
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

  int x = int(index % width);
  int y = int(index / width);

  float sum = 0.0F;
  for (int q = 0; q < filter_size; ++q) {
    int q_prime = height - filter_size / 2 + y + q;
    for (int p = 0; p < filter_size; ++p) {
      int p_prime = width - filter_size / 2 + x + p;
      if (pbp[twoToOne(p_prime, q_prime, width, height)] != 0) {
        sum += precomputed[twoToOne(p, q, filter_size, filter_size)];
      }
    }
  }

  filter_out[index] = sum;
}
