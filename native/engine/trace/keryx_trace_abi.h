// What the engine glue (host, keryx_llama.cpp) hands the trace ops that run inside ggml-cuda.
// A capture node is a GGML_OP_CUSTOM whose `fun` is keryx_trace_cpu_fun and whose `userdata`
// points at one keryx_trace_op; the op outlives every graph that names it, so a CUDA graph that
// captured its pointer stays valid. Plain C: both sides include it.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KERYX_TRACE_MAX_DEVICES 2
#define KERYX_TRACE_HASH 32

// What a capture node hashes into its leaf, one message per ubatch row it covers.
enum keryx_trace_kind {
    // Rows of an f32 tensor: Y of a product (after its bias or GLU), a residual, router scores.
    // With a second source (MUL_MAT_ID ids, i32 [n_used, rows]) each message is the row
    // followed by that row's ids: Y of an expert product.
    KERYX_TRACE_F32 = 1,
    // The Q8_1 input of a product, quantised again by llama's own quantiser from the f32 source
    // the product read (src0 of the node), in `layout`. `per_leaf` source rows make one leaf
    // (the n_used inputs of a down-experts product); otherwise one row per leaf.
    KERYX_TRACE_X = 2,
    // Sequence leaves, streamed over all positions (node seq.rs). The source is the node that
    // just wrote or used them; the op reads that node's own sources.
    // K or V cache rows of one attention layer, read from the cache after SET_ROWS wrote them
    // (sources: the SET_ROWS node, its cell indices).
    KERYX_TRACE_KV = 3,
    // Softmax mass m + ln(l) per position and query head of one attention layer, over the same
    // cache and mask FLASH_ATTN_EXT used (sources: the FA node, q, k, mask).
    KERYX_TRACE_LSE = 4,
    // Records and state checkpoints of one recurrent layer (sources: the GATED_DELTA_NET node,
    // q, k, v, g, beta).
    KERYX_TRACE_RECUR = 5,
    // A decode ubatch's captures only stage their bytes (in the device's defer arena); this node,
    // after the last capture of a run of layers on one card, hashes all of the run's messages in
    // one launch and appends all its sequence rows in one call.
    KERYX_TRACE_FLUSH = 6,
};

// The longest message a decode capture stages; a longer one (the logits) is hashed where it is
// captured, by many blocks, rather than by one flush block.
#define KERYX_DEFER_MAX_MSG (64u * 1024u)

// A staged message: bytes [off, off + len) of the defer arena are the leaf at `table` (row *
// n_leaves + leaf) of the leaf table.
struct keryx_trace_msg {
    uint64_t off;
    uint32_t len;
    uint32_t table;
};

// Q8_1 layouts, as the node numbers them (verifiable-inference q8.rs).
enum keryx_trace_layout {
    KERYX_TRACE_MMVQ = 0,
    KERYX_TRACE_MMQ_D4 = 1,
    KERYX_TRACE_MMQ_DS4 = 2,
};

// Per context, per device: where the leaf hashes of the running ubatch go.
struct keryx_trace_table {
    uint8_t* hashes;   // device: [row_cap][n_leaves][32]
    int device;        // CUDA ordinal, -1 = unused slot
};

// Per context, per device: the streamed sequence leaves and what the ubatch writes them from.
struct keryx_trace_seq {
    void * streams;        // device: kb3::stream_state[n_seq_leaves]; a leaf streams on one card
    int32_t * pos0;        // device: position of the ubatch's first row
    float * ckpt;          // device: recurrent states the GDN kernel copies out, [slot][head][hd*hd]
    uint64_t ckpt_bytes;
    int device;            // CUDA ordinal, -1 = unused slot
    // Replay for an opening: the bytes of chosen leaves, copied out as they are hashed. Offsets
    // are -1 except for a wanted leaf, so outside a replay every copy is a no-op and the graph
    // the card runs is the same.
    // The arenas are allocated for a replay only; the kernels read their address from these
    // device slots, so a captured graph keeps working.
    int64_t * pos_off;     // device: [row_cap][n_leaves] byte offset in the arena of that row's leaf
    uint8_t ** arena;      // device slot: the arena
    int64_t * seq_off;     // device: [n_seq_leaves] byte offset in the seq arena of the whole leaf
    uint8_t ** seq_arena;  // device slot: the seq arena
    // Decode staging (KERYX_TRACE_FLUSH): the arena, the static message table and the append
    // descriptors of the decode graph.
    uint8_t * defer;
    uint64_t defer_cap;
    struct keryx_trace_msg * msgs;
    uint32_t msgs_cap;
    void * descs;          // kb3::append_desc[descs_cap]
    uint32_t descs_cap;
};

struct keryx_trace_state {
    uint32_t n_leaves;
    uint32_t row_cap;  // rows the tables hold (the ubatch size)
    uint32_t n_seq_leaves;
    struct keryx_trace_table tables[KERYX_TRACE_MAX_DEVICES];
    struct keryx_trace_seq seq[KERYX_TRACE_MAX_DEVICES];
};

struct keryx_trace_op {
    uint32_t magic;            // KERYX_TRACE_MAGIC
    uint32_t kind;             // keryx_trace_kind
    uint32_t leaf;             // index in the position tree
    uint32_t layout;           // KERYX_TRACE_X: keryx_trace_layout
    int32_t  wtype;            // KERYX_TRACE_X: ggml type of the product's weight
    uint32_t per_leaf;         // KERYX_TRACE_X: source rows per leaf (1, n_used, or n_head)
    uint32_t alias;            // KERYX_TRACE_X decode: 1 = an X capture of the same source stages its bytes
    // F32 and X: 2 = the source is [k, rows, heads] and one leaf holds every head of a row
    // (ROLE_HEADS: the per-head products of a latent attention layer); 0 = rows in order.
    uint32_t heads_dim;
    // Table row of the first message: 0, or the first output row when the source holds only
    // the rows that produce logits (the LM head, a trimmed last layer).
    uint32_t row0;
    // Sequence kinds: leaf of head h = leaf + h * leaf_step (node seq.rs), for n_heads heads of
    // head_dim values; group = query heads per kv head (LSE); interval and recur_kind (RECUR).
    uint32_t leaf_step;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t group;
    uint32_t interval;
    uint32_t recur_kind;   // 0 GDA, 1 KDA
    // Decode staging: >= 0 = the op writes its messages (or its sequence rows) at this offset of
    // the defer arena instead of hashing them; desc = its first append descriptor (sequence kinds).
    // FLUSH: msgs [first, first + count) and descriptors [desc, desc + n_descs), staged by the
    // n_members ops in members; it runs only if each of them ran on its card.
    int64_t defer_off;
    uint32_t desc;
    uint32_t first;
    uint32_t count;
    uint32_t n_descs;
    struct keryx_trace_state* state;
    struct keryx_trace_op* const* members;
    uint32_t n_members;
    // The card the op last ran on, written when its node runs on the host (graph capture).
    int32_t ran_on;
};

// What the patched GATED_DELTA_NET kernel needs to copy out the states the trace checkpoints:
// after position p it copies the state of every head when (p + 1) % interval == 0, into
// ckpt + ((p + 1) / interval - first) * heads * hd * hd, first = (pos0 + interval) / interval.
// The trace stores a pointer to it in the node's op_params at KERYX_TRACE_GDN_PARAM.
struct keryx_trace_gdn {
    uint32_t magic;            // KERYX_TRACE_MAGIC
    uint32_t interval;
    struct keryx_trace_state* state;
};

#define KERYX_TRACE_GDN_PARAM 8  // byte offset in op_params, past the op's own K

#define KERYX_TRACE_MAGIC 0x5254584bu  // "KXTR"

// Forgets the Q8_1 rows decode X captures lent to the products on this thread; the engine calls
// it after every traced ubatch, so no other graph can read them.
void keryx_trace_forget_q8_1(void);

#ifdef __cplusplus
}
#endif
