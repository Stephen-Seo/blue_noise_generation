__kernel void do_filter(
        __global float *filter_out, __global float *precomputed,
        __global int *pbp, int width, int height, int filter_size) {
    int i = get_global_id(0);
    if(i < 0 || i >= width * height) {
        return;
    }
    int x = i % width;
    int y = i / width;

    float sum = 0.0f;
    for(int q = 0; q < filter_size; ++q) {
        int q_prime = (height + filter_size / 2 + y - q) % height;
        for(int p = 0; p < filter_size; ++p) {
            int p_prime = (width + filter_size / 2 + x - p) % width;
            if(pbp[p_prime + q_prime * width] != 0) {
                sum += precomputed[p + q * filter_size];
            }
        }
    }

    filter_out[i] = sum;
}

// vim: syntax=c
