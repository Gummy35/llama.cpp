#include "debug.h"

#include "common.h"
#include "log.h"

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

struct common_moe_cb_user_data {
    std::vector<uint8_t> data;
};

static int parse_layer_number(const char * tensor_name) {
    // Find the last occurrence of '-' in the tensor name
    const char * last_dash = strrchr(tensor_name, '-');
    if (last_dash == nullptr) {
        return -1; // Invalid format
    }
    
    // Extract the layer number after the last dash
    int layer = 0;
    const char * p = last_dash + 1;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    
    // Check if we've consumed all digits after the dash
    if (*p != '\0') {
        return -1; // Invalid format
    }
    
    return layer;
}

static bool common_moe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_moe_cb_user_data *) user_data;

    constexpr const char * prefix = "ffn_moe_topk";
    constexpr size_t prefix_len = sizeof("ffn_moe_topk") - 1;

    // Scheduler asks whether we want this tensor.
    if (ask) {
        return strncmp(t->name, prefix, prefix_len) == 0;
    }

    // Should never happen, but keep the callback robust.
    if (strncmp(t->name, prefix, prefix_len) != 0) {
        return true;
    }

    const int layer = parse_layer_number(t->name);
    if (layer < 0) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_I32);

    // Retrieve tensor data
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    uint8_t * data;

    if (is_host) {
        data = static_cast<uint8_t *>(t->data);
    } else {
        const size_t nbytes = ggml_nbytes(t);
        cb_data->data.resize(nbytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, nbytes);
        data = cb_data->data.data();
    }

    // Tensor shape:
    // ne[0] = n_expert_used
    // ne[1] = n_tokens
    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const char * base = reinterpret_cast<const char *>(data);

    for (int64_t token = 0; token < n_tokens; ++token) {
        LOG("MOE %d %lld", layer, (long long) token);
        for (int64_t expert = 0; expert < n_expert_used; ++expert) {
            const int32_t value =
                *reinterpret_cast<const int32_t *>(
                    base +
                    token  * t->nb[1] +
                    expert * t->nb[0]);

            LOG(" %d", value);
        }
        LOG("\n");
    }
    return true;
}

void common_set_moe_callback(common_params & params) {
    if (params.log_moe) {
        static common_moe_cb_user_data cb_data;  // Static to keep it alive during model lifetime
        params.cb_eval = common_moe_cb_eval;
        params.cb_eval_user_data = &cb_data;
    }
}