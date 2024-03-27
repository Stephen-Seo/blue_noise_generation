#version 450

struct FloatAndIndex {
  float value;
  int pbp;
  int idx;
};

layout(binding = 0) readonly buffer MaxIn { FloatAndIndex max_in[]; };
layout(binding = 1) readonly buffer MinIn { FloatAndIndex min_in[]; };
layout(binding = 2) writeonly buffer MaxOut { FloatAndIndex max_out[]; };
layout(binding = 3) writeonly buffer MinOut { FloatAndIndex min_out[]; };

layout(binding = 4) readonly buffer State { int size; };

layout(local_size_x = 256) in;

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= (size + 1) / 2) {
    return;
  }

  if (index * 2 + 1 < size) {
    if (max_in[index * 2].pbp != 0 && max_in[index * 2 + 1].pbp != 0) {
      if (max_in[index * 2].value > max_in[index * 2 + 1].value) {
        max_out[index] = max_in[index * 2];
      } else {
        max_out[index] = max_in[index * 2 + 1];
      }
    } else if (max_in[index * 2].pbp != 0 && max_in[index * 2 + 1].pbp == 0) {
      max_out[index] = max_in[index * 2];
    } else {
      max_out[index] = max_in[index * 2 + 1];
    }

    if (min_in[index * 2].pbp == 0 && min_in[index * 2 + 1].pbp == 0) {
      if (min_in[index * 2].value < min_in[index * 2 + 1].value) {
        min_out[index] = min_in[index * 2];
      } else {
        min_out[index] = min_in[index * 2 + 1];
      }
    } else if (min_in[index * 2].pbp == 0 && min_in[index * 2 + 1].pbp != 0) {
      min_out[index] = min_in[index * 2];
    } else {
      min_out[index] = min_in[index * 2 + 1];
    }
  } else {
    max_out[index] = max_in[index * 2];
    min_out[index] = min_in[index * 2];
  }
}
