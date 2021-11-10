int twoToOne(x, y, width, height) {
    while(x < 0) {
        x += width;
    }
    while(y < 0) {
        y += height;
    }
    x = x % width;
    y = y % height;
    return x + y * width;
}

//float gaussian(float x, float y) {
//    return exp(-(x*x + y*y) / (1.5F * 1.5F * 2.0F));
//}

__kernel void do_filter(
        __global float *filter_out, __global const float *precomputed,
        __global const int *pbp, const int width, const int height,
        const int filter_size) {
    int i = get_global_id(0);
    if(i < 0 || i >= width * height) {
        return;
    }

    int x = i % width;
    int y = i / width;

    float sum = 0.0F;
    for(int q = 0; q < filter_size; ++q) {
        int q_prime = height - filter_size / 2 + y + q;
        for(int p = 0; p < filter_size; ++p) {
            int p_prime = width - filter_size / 2 + x + p;
            if(pbp[twoToOne(p_prime, q_prime, width, height)] != 0) {
                sum += precomputed[twoToOne(p, q, filter_size, filter_size)];
                //sum += gaussian(p - filter_size / 2.0F + 0.5F, q - filter_size / 2.0F + 0.5F);
            }
        }
    }

    filter_out[i] = sum;
}

// vim: syntax=c
