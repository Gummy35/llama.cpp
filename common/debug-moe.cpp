#include "debug.h"

#include "common.h"
#include "log.h"

#include <cstring>
#include <vector>

struct common_moe_cb_user_data {
    // Temporary buffer used when the tensor is stored on the GPU.
    std::vector<uint8_t> data;

    // Write the log header only once.
    bool header_written = false;
};

/**
 * Extract the transformer layer number from a tensor name.
 *
 * Expected format:
 *     ffn_moe_topk-<layer>
 *
 * Returns:
 *     layer index
 *     -1 if the name does not match the expected format.
 */
static int parse_layer_number(const char * tensor_name) {
    const char * last_dash = strrchr(tensor_name, '-');

    if (last_dash == nullptr) {
        return -1;
    }

    int layer = 0;
    const char * p = last_dash + 1;

    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        ++p;
    }

    return (*p == '\0') ? layer : -1;
}

static bool common_moe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_moe_cb_user_data *) user_data;

    constexpr const char * prefix = "ffn_moe_topk";
    constexpr size_t prefix_len = sizeof("ffn_moe_topk") - 1;

    //
    // Scheduler asks whether we are interested in this tensor.
    //
    if (ask) {
        return strncmp(t->name, prefix, prefix_len) == 0;
    }

    //
    // Ignore every other tensor.
    //
    if (strncmp(t->name, prefix, prefix_len) != 0) {
        return true;
    }

    const int layer = parse_layer_number(t->name);

    if (layer < 0) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_I32);

    //
    // Retrieve tensor data.
    // If the tensor is on the GPU, copy it to a temporary host buffer.
    //
    uint8_t * data;

    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<uint8_t *>(t->data);
    } else {
        const size_t nbytes = ggml_nbytes(t);

        cb_data->data.resize(nbytes);

        ggml_backend_tensor_get(
            t,
            cb_data->data.data(),
            0,
            nbytes);

        data = cb_data->data.data();
    }

    //
    // Tensor layout:
    //
    // ne[0] = number of selected experts (top-k)
    // ne[1] = number of processed tokens
    //
    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];

    const char * base = reinterpret_cast<const char *>(data);

    //
    // Header written once at the beginning of the log.
    //
    if (!cb_data->header_written) {

        cb_data->header_written = true;

        LOG("# ------------------------------------------------------------\n");
        LOG("# MoE routing log\n");
        LOG("# One line = one transformer layer evaluated for one token.\n");
        LOG("#\n");
        LOG("# Format:\n");
        LOG("#   MOE <layer> <token> <expert0> ... <expertN>\n");
        LOG("#\n");
        LOG("# Columns:\n");
        LOG("#   layer token");

        for (int64_t i = 0; i < n_expert_used; ++i) {
            LOG(" expert%lld", (long long) i);
        }

        LOG("\n");
        LOG("# ------------------------------------------------------------\n");
    }

    //
    // Output one line per token.
    //
    // Token numbering is local to the current callback.
    // During prompt ingestion, token ranges typically span:
    //
    //   layer 0 : token 0..N-1
    //   layer 1 : token 0..N-1
    //   ...
    //
    // During autoregressive decoding, n_tokens is usually equal to 1,
    // therefore token is always 0.
    //
    // This is sufficient for statistical analysis of expert usage.
    //
    for (int64_t token = 0; token < n_tokens; ++token) {

        LOG("MOE %d %lld",
            layer,
            (long long) token);

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

    if (!params.log_moe) {
        return;
    }

    // Static so that the temporary buffer survives during the whole
    // lifetime of the inference context.
    static common_moe_cb_user_data cb_data;

    params.cb_eval = common_moe_cb_eval;
    params.cb_eval_user_data = &cb_data;
}