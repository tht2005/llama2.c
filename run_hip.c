#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <hip/hip_runtime.h>

// Wrap every HIP call so failures are loud instead of silent.
#define HIP_CHECK(cmd) do {                                   \
    hipError_t e = (cmd);                                     \
    if (e != hipSuccess) {                                    \
        fprintf(stderr, "HIP error %s:%d: %s\n",              \
                __FILE__, __LINE__, hipGetErrorString(e));    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

// All pointers below are DEVICE (GPU) pointers.
typedef struct {
    float *token_embedding_table; // (vocab_size, dim)
    float *rms_att_weight;        // (layer, dim)
    float *wq, *wk, *wv, *wo;      // attention projections
    float *rms_ffn_weight;        // (layer, dim)
    float *w1, *w2, *w3;           // ffn
    float *rms_final_weight;      // (dim,)
    float *wcls;                  // classifier (vocab_size, dim)
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2;   // (seq_len, dim)
    float *hb, *hb2;       // (seq_len, hidden_dim)
    float *q;              // (seq_len, dim)
    float *att;            // (n_heads, seq_len * seq_len)
    float *logits_dev;     // (vocab_size,) on device
    float *logits;         // (vocab_size,) on host
    float *key_cache, *value_cache;   // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float *weights_dev;    // one contiguous device blob backing all weights
} Transformer;

static void map_weights_device(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;   ptr += (unsigned long long)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;          ptr += n_layers * p->dim;
    w->wq = ptr;                      ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                      ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                      ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                      ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;          ptr += n_layers * p->dim;
    w->w1 = ptr;                      ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                      ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                      ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;        ptr += p->dim;
    ptr += p->seq_len * head_size / 2;  // skip legacy freq_cis_real
    ptr += p->seq_len * head_size / 2;  // skip legacy freq_cis_imag
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    HIP_CHECK(hipMalloc(&s->x,   p->seq_len * p->dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->xb,  p->seq_len * p->dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->xb2, p->seq_len * p->dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->hb,  p->seq_len * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->hb2, p->seq_len * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->q,   p->seq_len * p->dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->att, (size_t)p->n_heads * p->seq_len * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->logits_dev, p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->key_cache,   (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&s->value_cache, (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    s->logits = (float *)malloc(p->vocab_size * sizeof(float));
    if (!s->logits) { fprintf(stderr, "host malloc failed\n"); exit(EXIT_FAILURE); }
}

static void free_run_state(RunState *s) {
    hipFree(s->x); hipFree(s->xb); hipFree(s->xb2);
    hipFree(s->hb); hipFree(s->hb2); hipFree(s->q);
    hipFree(s->att); hipFree(s->logits_dev);
    hipFree(s->key_cache); hipFree(s->value_cache);
    free(s->logits);
}

static void build_transformer(Transformer *t, char *checkpoint_path) {
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    int fd = open(checkpoint_path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    float *data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    size_t header_floats = sizeof(Config) / sizeof(float);
    size_t weight_floats = file_size / sizeof(float) - header_floats;
    HIP_CHECK(hipMalloc(&t->weights_dev, weight_floats * sizeof(float)));
    HIP_CHECK(hipMemcpy(t->weights_dev, data + header_floats,
                        weight_floats * sizeof(float), hipMemcpyHostToDevice));
    munmap(data, file_size);
    close(fd);

    map_weights_device(&t->weights, &t->config, t->weights_dev, shared_weights);
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer *t) {
    hipFree(t->weights_dev);
    free_run_state(&t->state);
}

#define BLK 256

// matmul: W(d,n) @ x(n) -> xout(d). The hot loop of the whole model.
// TODO: one thread per row i; xout[i] = sum over j in [0,n) of  w[i*n + j] * x[j]
//       (plain sequential loop -> matches the CPU's add order, keeps logits close).
// DUMMY: passes the input through (wrong, but keeps the pipeline running).
__global__ void matmul_kernel(float *xout, const float *x, const float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float sum = 0;
    for (int j = 0; j < n; ++j) {
        sum += w[i*n + j] * x[j];
    }
    xout[i] = sum;
}

// W(d, n) @ x(B, n) -> xout(B, d)
__global__ void matmul_batched_kernel(float *xout, const float *x, const float *w, int n, int d, int B) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d || b >= B) return;

    const float *x_row = x + b*n;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        sum += w[i*n + j] * x_row[j];
    }
    xout[b*d + i] = sum;
}

// rmsnorm: normalize x (RMS over `size`) and scale by `weight`. Launched <<<1,BLK>>>.
// TODO: 1) each thread sums x[j]*x[j] over a strided range; block-reduce in __shared__
//          memory (halving loop);  2) scale = 1/sqrtf(sumsq/size + 1e-5f);
//       3) o[j] = weight[j] * (scale * x[j]).  __syncthreads() between phases.
// DUMMY: copies x through unchanged.
__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight, int size) {
    __shared__ float sdata[BLK];
    const int tid = threadIdx.x;
    
    float strided_sum = 0;
    for (int j = tid; j < size; j += blockDim.x) {
        strided_sum += x[j] * x[j];
    }
    sdata[tid] = strided_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float scale;
    if (tid == 0) {
        float sumsq = sdata[0];
        scale = 1.0f / sqrtf(sumsq / size + 1e-5f);
    }
    __syncthreads();

    for (int j = tid; j < size; j += blockDim.x) {
        o[j] = weight[j] * (scale * x[j]);
    }
}

__global__ void rmsnorm_batched_kernel(float *o, const float *x, const float *weight, int size, int B) {
    const int b = blockIdx.x;
    if (b >= B) return;

    __shared__ float sdata[BLK];
    const int tid = threadIdx.x;

    const float *x_row = x + b*size;
    float *o_row = o + b*size;

    float strided_sum = 0.0f;
    for (int j = tid; j < size; j += blockDim.x) {
        strided_sum += x_row[j] * x_row[j];
    }
    sdata[tid] = strided_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float scale;
    if (tid == 0) {
        scale = 1.0f / sqrtf(sdata[0] / size + 1e-5f);
    }
    __syncthreads();

    for (int j = tid; j < size; j += blockDim.x) {
        o_row[j] = weight[j] * (scale * x_row[j]);
    }
}

// RoPE: rotate q (and k while idx<kv_dim) by angle pos*freq to encode position.
// TODO: idx = tid*2 (guard idx>=dim); head_dim = idx % head_size;
//       freq = 1/powf(10000,head_dim/(float)head_size); fcr=cosf(pos*freq); fci=sinf(pos*freq);
//       rotate (q[idx],q[idx+1]); if idx<kv_dim rotate (k[idx],k[idx+1]) the same way.
// DUMMY: leaves q and k unchanged.
__global__ void rope_kernel(float *q, float *k, int pos, int dim, int kv_dim, int head_size) {
    const int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = g_tid << 1;
    if (idx >= dim) return;
    const int head_dim = idx % head_size;

    const float freq = 1.0f / powf(10000, head_dim / (float)head_size);
    const float fcr = cosf(pos * freq);
    const float fci = sinf(pos * freq);

    // rotate q
    float q0 = q[idx];
    float q1 = q[idx + 1];
    q[idx] = q0 * fcr - q1 * fci;
    q[idx + 1] = q0 * fci + q1 * fcr;

    // rotate k
    if (idx < kv_dim) {
        float k0 = k[idx];
        float k1 = k[idx + 1];
        k[idx] = k0 * fcr - k1 * fci;
        k[idx + 1] = k0 * fci + k1 * fcr;
    }
}

__global__ void rope_batched_kernel(float *q, float *k, int pos, int dim, int kv_dim, int head_size, int B) {
    const int b = blockIdx.y;
    const int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = g_tid << 1;
    if (b >= B || idx >= dim) return;

    const int head_dim = idx % head_size;
    
    const int cur_pos = pos + b;

    const float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    const float fcr = cosf(cur_pos * freq);
    const float fci = sinf(cur_pos * freq);

    float *q_row = q + b * dim;
    float q0 = q_row[idx];
    float q1 = q_row[idx + 1];
    q_row[idx] = q0 * fcr - q1 * fci;
    q_row[idx + 1] = q0 * fci + q1 * fcr;

    if (idx < kv_dim) {
        float *k_row = k + b * kv_dim;
        float k0 = k_row[idx];
        float k1 = k_row[idx + 1];
        k_row[idx] = k0 * fcr - k1 * fci;
        k_row[idx + 1] = k0 * fci + k1 * fcr;
    }
}

// attention: ONE BLOCK PER HEAD (blockIdx.x = h). scores -> softmax -> weighted sum -> xb.
// TODO: q=q_all+h*head_size; att=att_all+h*seq_len; kbase=layer_k+(h/kv_mul)*head_size (vbase too).
//       1) scores: att[t] = (q . k_t)/sqrtf(head_size) for t in [0..pos]  (k_t = kbase + t*kv_dim)
//       2) softmax over att[0..pos]: block-reduce MAX, subtract & expf, block-reduce SUM
//          (reuse rmsnorm's __shared__ reduction; __syncthreads() between phases!)
//       3) xb[i] = (sum_t att_exp[t] * v_t[i]) / sum,  xb = xb_all + h*head_size, i in [0..head_size)
// DUMMY: leaves xb unchanged (it still holds the rmsnorm output) — i.e. skips attention.
__global__ void attention_kernel(float *att_all, const float *q_all,
                                  const float *layer_k, const float *layer_v,
                                  float *xb_all, int pos, int head_size,
                                  int kv_dim, int kv_mul, int seq_len) {
    __shared__ float sdata[BLK];

    const int h = blockIdx.x;
    const int tid = threadIdx.x;

    const float *q = q_all + h * head_size;
    float *xb = xb_all + h * head_size;
    
    float *att = att_all + h * seq_len;

    const int kv_head = h / kv_mul;
    const float *kbase = layer_k + kv_head * head_size;
    const float *vbase = layer_v + kv_head * head_size;

    float scale = 1.0f / sqrtf((float)head_size);

    for (int t = tid; t <= pos; t += blockDim.x) {
        const float *k_t = kbase + t * kv_dim;
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) {
            score += q[i] * k_t[i];
        }
        att[t] = score * scale;
    }
    __syncthreads();

    float local_max = -1e20f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        if (att[t] > local_max) {
            local_max = att[t];
        }
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    __shared__ float max_val;
    if (tid == 0) max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float exp_score = expf(att[t] - max_val);
        att[t] = exp_score;
        local_sum += exp_score;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float sum_total;
    if (tid == 0) sum_total = sdata[0];
    __syncthreads();

    for (int i = tid; i < head_size; i += blockDim.x) {
        float val_acc = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float* v_t = vbase + t * kv_dim;
            val_acc += att[t] * v_t[i];
        }
        xb[i] = val_acc / sum_total;
    }
}

__global__ void attention_prefill_kernel(float *att_all, const float *q_all,
                                         const float *layer_k, const float *layer_v,
                                         float *xb_all, int num_rows, int head_size,
                                         int kv_dim, int kv_mul, int seq_len) {
    const int h = blockIdx.x; 
    const int tid = threadIdx.x;

    const float *q_head = q_all + h * head_size;
    float *xb_head = xb_all + h * head_size;
    float *att_head = att_all + h * seq_len * seq_len;

    int kv_head = h / kv_mul;
    const float *kbase = layer_k + kv_head * head_size;
    const float *vbase = layer_v + kv_head * head_size;
    float scale = 1.0f / sqrtf((float)head_size);

    for (int r = 0; r < num_rows; ++r) {
        const float *q = q_head + r * (gridDim.x * head_size);
        float *att = att_head + r * seq_len;

        for (int t = tid; t < num_rows; t += blockDim.x) {
            if (t <= r) {
                const float *k_t = kbase + t * kv_dim;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += q[i] * k_t[i];
                att[t] = score * scale;
            } else {
                att[t] = -1e20f; 
            }
        }
        __syncthreads();

        __shared__ float sdata[BLK];
        float local_max = -1e20f;
        for (int t = tid; t <= r; t += blockDim.x) {
            if (att[t] > local_max) local_max = att[t];
        }
        sdata[tid] = local_max;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && sdata[tid + stride] > sdata[tid]) sdata[tid] = sdata[tid + stride];
            __syncthreads();
        }
        __shared__ float max_val;
        if (tid == 0) max_val = sdata[0];
        __syncthreads();

        float local_sum = 0.0f;
        for (int t = tid; t < num_rows; t += blockDim.x) {
            if (t <= r) {
                float exp_score = expf(att[t] - max_val);
                att[t] = exp_score;
                local_sum += exp_score;
            } else {
                att[t] = 0.0f;
            }
        }
        sdata[tid] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        __shared__ float sum_total;
        if (tid == 0) sum_total = sdata[0];
        __syncthreads();

        float *xb = xb_head + r * (gridDim.x * head_size);
        for (int i = tid; i < head_size; i += blockDim.x) {
            float val_acc = 0.0f;
            for (int t = 0; t <= r; ++t) {
                const float* v_t = vbase + t * kv_dim;
                val_acc += att[t] * v_t[i];
            }
            xb[i] = val_acc / sum_total;
        }
        __syncthreads();
    }
}

// residual: x[i] += y[i].   DUMMY: leaves x as-is (drops the residual add).
__global__ void residual_kernel(float *x, const float *y, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    x[i] += y[i];
}

__global__ void residual_batched_kernel(float *x, const float *y, int size, int num_rows) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= size || row >= num_rows) return;
    int idx = row * size + col;
    x[idx] += y[idx];
}

// swiglu: hb[i] = silu(hb[i]) * hb2[i], with silu(v)=v/(1+expf(-v)).
// DUMMY: leaves hb as-is.
__global__ void swiglu_kernel(float *hb, const float *hb2, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float silu = hb[i] / (1 + expf(-hb[i]));
    hb[i] = silu * hb2[i];
}

__global__ void swiglu_batched_kernel(float *hb, const float *hb2, int size, int num_rows) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= size || row >= num_rows) return;
    int idx = row * size + col;
    float val = hb[idx];
    float silu = val / (1.0f + expf(-val));
    hb[idx] = silu * hb2[idx];
}

static inline int grid(int n) { return (n + BLK - 1) / BLK; }  // blocks to cover n

static void forward_prefill(Transformer *transformer, const int *tokens, int prompt_len) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    for (int i = 0; i < prompt_len; i++) {
        HIP_CHECK(hipMemcpy(s->x + (size_t)i * dim, w->token_embedding_table + (size_t)tokens[i] * dim,
                            dim * sizeof(float), hipMemcpyDeviceToDevice));
    }

    dim3 block_dim(BLK);
    dim3 grid_dim_dim(grid(dim), prompt_len);
    dim3 grid_dim_kv(grid(kv_dim), prompt_len);
    dim3 grid_dim_hidden(grid(hidden_dim), prompt_len);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_batched_kernel<<<prompt_len, BLK>>>(s->xb, s->x, w->rms_att_weight + l * dim, dim, prompt_len);

        size_t loff = (size_t)l * p->seq_len * kv_dim;
        float *k_dest = s->key_cache + loff;
        float *v_dest = s->value_cache + loff;

        matmul_batched_kernel<<<grid_dim_dim, block_dim>>>(s->q, s->xb, w->wq + l * dim * dim, dim, dim, prompt_len);
        matmul_batched_kernel<<<grid_dim_kv, block_dim>>>(k_dest, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim, prompt_len);
        matmul_batched_kernel<<<grid_dim_kv, block_dim>>>(v_dest, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim, prompt_len);

        rope_batched_kernel<<<grid_dim_dim, block_dim>>>(s->q, k_dest, 0, dim, kv_dim, head_size, prompt_len);

        attention_prefill_kernel<<<p->n_heads, BLK>>>(s->att, s->q, k_dest, v_dest, s->xb, prompt_len, head_size, kv_dim, kv_mul, p->seq_len);

        matmul_batched_kernel<<<grid_dim_dim, block_dim>>>(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim, prompt_len);
        residual_batched_kernel<<<grid_dim_dim, block_dim>>>(s->x, s->xb2, dim, prompt_len);

        rmsnorm_batched_kernel<<<prompt_len, BLK>>>(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, prompt_len);

        matmul_batched_kernel<<<grid_dim_hidden, block_dim>>>(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, prompt_len);
        matmul_batched_kernel<<<grid_dim_hidden, block_dim>>>(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim, prompt_len);
        swiglu_batched_kernel<<<grid_dim_hidden, block_dim>>>(s->hb, s->hb2, hidden_dim, prompt_len);
        matmul_batched_kernel<<<grid_dim_dim, block_dim>>>(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, prompt_len);

        residual_batched_kernel<<<grid_dim_dim, block_dim>>>(s->x, s->xb, dim, prompt_len);
    }

    float *last_x_row = s->x + (size_t)(prompt_len - 1) * dim;
    rmsnorm_batched_kernel<<<1, BLK>>>(s->xb, last_x_row, w->rms_final_weight, dim, 1);
    matmul_batched_kernel<<<grid(p->vocab_size), BLK>>>(s->logits_dev, s->xb, w->wcls, dim, p->vocab_size, 1);
    HIP_CHECK(hipMemcpy(s->logits, s->logits_dev, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
}

static float *forward_decode(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    HIP_CHECK(hipMemcpy(s->x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float), hipMemcpyDeviceToDevice));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_batched_kernel<<<1, BLK>>>(s->xb, s->x, w->rms_att_weight + l * dim, dim, 1);

        size_t loff = (size_t)l * p->seq_len * kv_dim;
        float *k = s->key_cache + loff + (size_t)pos * kv_dim;
        float *v = s->value_cache + loff + (size_t)pos * kv_dim;

        matmul_batched_kernel<<<grid(dim), BLK>>>(s->q, s->xb, w->wq + l * dim * dim, dim, dim, 1);
        matmul_batched_kernel<<<grid(kv_dim), BLK>>>(k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim, 1);
        matmul_batched_kernel<<<grid(kv_dim), BLK>>>(v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim, 1);

        rope_batched_kernel<<<grid(dim / 2), BLK>>>(s->q, k, pos, dim, kv_dim, head_size, 1);

        attention_kernel<<<p->n_heads, BLK>>>(s->att, s->q, s->key_cache + loff, s->value_cache + loff, s->xb, pos, head_size, kv_dim, kv_mul, p->seq_len);

        matmul_batched_kernel<<<grid(dim), BLK>>>(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim, 1);
        residual_batched_kernel<<<grid(dim), BLK>>>(s->x, s->xb2, dim, 1);

        rmsnorm_batched_kernel<<<1, BLK>>>(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, 1);

        matmul_batched_kernel<<<grid(hidden_dim), BLK>>>(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, 1);
        matmul_batched_kernel<<<grid(hidden_dim), BLK>>>(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim, 1);
        swiglu_batched_kernel<<<grid(hidden_dim), BLK>>>(s->hb, s->hb2, hidden_dim, 1);
        matmul_batched_kernel<<<grid(dim), BLK>>>(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, 1);

        residual_batched_kernel<<<grid(dim), BLK>>>(s->x, s->xb, dim, 1);
    }

    rmsnorm_batched_kernel<<<1, BLK>>>(s->x, s->x, w->rms_final_weight, dim, 1);
    matmul_batched_kernel<<<grid(p->vocab_size), BLK>>>(s->logits_dev, s->x, w->wcls, dim, p->vocab_size, 1);

    HIP_CHECK(hipMemcpy(s->logits, s->logits_dev, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return s->logits;
}

static float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // token embedding -> x (device-to-device copy)
    HIP_CHECK(hipMemcpy(s->x, w->token_embedding_table + (size_t)token * dim,
                        dim * sizeof(float), hipMemcpyDeviceToDevice));

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm_kernel<<<1, BLK>>>(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        size_t loff = l * p->seq_len * kv_dim;
        float *k = s->key_cache + loff + (size_t)pos * kv_dim;
        float *v = s->value_cache + loff + (size_t)pos * kv_dim;

        matmul_kernel<<<grid(dim), BLK>>>(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul_kernel<<<grid(kv_dim), BLK>>>(k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul_kernel<<<grid(kv_dim), BLK>>>(v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        rope_kernel<<<grid(dim / 2), BLK>>>(s->q, k, pos, dim, kv_dim, head_size);

        attention_kernel<<<p->n_heads, BLK>>>(s->att, s->q,
            s->key_cache + loff, s->value_cache + loff, s->xb,
            pos, head_size, kv_dim, kv_mul, p->seq_len);

        matmul_kernel<<<grid(dim), BLK>>>(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        residual_kernel<<<grid(dim), BLK>>>(s->x, s->xb2, dim);

        rmsnorm_kernel<<<1, BLK>>>(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        matmul_kernel<<<grid(hidden_dim), BLK>>>(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul_kernel<<<grid(hidden_dim), BLK>>>(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        swiglu_kernel<<<grid(hidden_dim), BLK>>>(s->hb, s->hb2, hidden_dim);
        matmul_kernel<<<grid(dim), BLK>>>(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        residual_kernel<<<grid(dim), BLK>>>(s->x, s->xb, dim);
    }

    rmsnorm_kernel<<<1, BLK>>>(s->x, s->x, w->rms_final_weight, dim);
    matmul_kernel<<<grid(p->vocab_size), BLK>>>(s->logits_dev, s->x, w->wcls, dim, p->vocab_size);

    HIP_CHECK(hipMemcpy(s->logits, s->logits_dev, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return s->logits;
}

typedef struct { char *str; int id; } TokenIndex;
typedef struct {
    char **vocab; float *vocab_scores; TokenIndex *sorted_vocab;
    int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) { t->byte_pieces[i*2] = (unsigned char)i; t->byte_pieces[i*2+1] = '\0'; }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

static void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
}

static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = (char *)t->byte_pieces + byte_val * 2;
    return piece;
}

static void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok; tok.str = str;
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

static void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else for (size_t i = 0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10; int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex *probindex; float temperature, topp; unsigned long long rng_state; } Sampler;

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}
static int sample_argmax(float *p, int n) {
    int mi = 0; float mp = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mp) { mi = i; mp = p[i]; }
    return mi;
}
static int sample_mult(float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
static int compare_prob(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *)a, *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}
static int sample_topp(float *p, int n, float topp, ProbIndex *pi, float coin) {
    int n0 = 0; const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (p[i] >= cutoff) { pi[n0].index = i; pi[n0].prob = p[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), compare_prob);
    float cum = 0.0f; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += pi[i].prob; if (cum > topp) { last = i; break; } }
    float r = coin * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) { cdf += pi[i].prob; if (r < cdf) return pi[i].index; }
    return pi[last].index;
}
static void build_sampler(Sampler *s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size; s->temperature = temperature; s->topp = topp; s->rng_state = seed;
    s->probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex));
}
static void free_sampler(Sampler *s) { free(s->probindex); }
static unsigned int random_u32(unsigned long long *st) {
    *st ^= *st >> 12; *st ^= *st << 25; *st ^= *st >> 27;
    return (*st * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *st) { return (random_u32(st) >> 8) / 16777216.0f; }
static int sample(Sampler *s, float *logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0 || s->topp >= 1) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

static long time_in_ms() {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static void generate_with_prefill_decode_split(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                     char *prompt, int steps) {
    char empty[] = "";
    if (prompt == NULL) prompt = empty;
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "expected >=1 prompt token\n"); exit(EXIT_FAILURE); }
    if (steps > transformer->config.seq_len) steps = transformer->config.seq_len;

    long start = time_in_ms();

    forward_prefill(transformer, prompt_tokens, num_prompt_tokens);
    int token = prompt_tokens[num_prompt_tokens - 1];
    int next = sample(sampler, transformer->state.logits);
    int pos = num_prompt_tokens;

    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); fflush(stdout);
    token = next;    

    while (pos < steps) {
        float *logits = forward_decode(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;
        if (next == 1) break;
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); fflush(stdout);
        token = next;
    }
    printf("\n");
    if (pos > num_prompt_tokens) {
        long end = time_in_ms();
        fprintf(stderr, "[prefill] achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);    
}

static void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                     char *prompt, int steps) {
    char empty[] = "";
    if (prompt == NULL) prompt = empty;
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "expected >=1 prompt token\n"); exit(EXIT_FAILURE); }
    if (steps > transformer->config.seq_len) steps = transformer->config.seq_len;

    long start = 0; int next; int token = prompt_tokens[0]; int pos = 0;
    while (pos < steps) {
        float *logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
        else next = sample(sampler, logits);
        pos++;
        if (next == 1) break;
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "[original] achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

static void error_usage() {
    fprintf(stderr, "Usage: runhip <checkpoint> [-t temp] [-p topp] [-s seed] [-n steps] [-i prompt] [-z tokenizer]\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = (char *)"tokenizer.bin";
    float temperature = 1.0f, topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    if (argc >= 2) checkpoint_path = argv[1]; else error_usage();
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) error_usage();
        switch (argv[i][1]) {
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            default: error_usage();
        }
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    generate_with_prefill_decode_split(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

