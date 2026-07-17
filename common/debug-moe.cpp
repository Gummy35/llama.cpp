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

    if (ask) {
        // Only interested in tensors whose name starts with "ffn_moe_topk"
        return strncmp(t->name, "ffn_moe_topk", 11) == 0;
    }

    // Process the actual data for ffn_moe_topk tensors
    if (strncmp(t->name, "ffn_moe_topk", 11) != 0) {
        return true;
    }

    // Parse layer number from tensor name like "ffn_moe_topk-5"
    int layer = parse_layer_number(t->name);
    if (layer < 0) {
        return true; // Skip invalid tensor names
    }

    // Get tensor data - use direct access when possible, otherwise fallback to ggml_backend_tensor_get
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    
    uint8_t * data_ptr;
    if (is_host) {
        data_ptr = (uint8_t *)t->data;
    } else {
        size_t n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
        data_ptr = cb_data->data.data();
    }

    // For MoE topk tensors, the data contains expert indices for each token
    // The tensor shape is [n_expert_used, n_tokens]
    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    
    // Assuming the tensor is I32 (int32) type which is standard for indices
    const int32_t * expert_indices = (const int32_t *) data_ptr;
    
    // Log each token with its expert indices
    for (int64_t i_token = 0; i_token < n_tokens; i_token++) {
        // Log exactly two experts per token as per example format
        if (n_expert_used >= 2) {
            int32_t expert0 = expert_indices[i_token * n_expert_used + 0];
            int32_t expert1 = expert_indices[i_token * n_expert_used + 1];
            LOG("MOE %d %ld %d %d\n", layer, i_token, expert0, expert1);
        } else if (n_expert_used >= 1) {
            // Log at least one expert
            int32_t expert0 = expert_indices[i_token * n_expert_used + 0];
            LOG("MOE %d %ld %d -1\n", layer, i_token, expert0);
        }
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