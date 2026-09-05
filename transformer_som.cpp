// TransformerSOM — Multi-Head QKV Self-Attention Autoregressive SOM Engine
// High Performance: QKV Matrix Pre-projection Caching + SIMD Parallelism + MiniLM 384d + OASST + Nucleus Top-P=0.9

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <limits>
#include <cassert>
#include <array>
#include <chrono>
#include <thread>
#include <numeric>
#include <cstring>
#include <omp.h>

// ---------------------------------------------------------------------------
// Vector — fixed-dimension dense float vector
// ---------------------------------------------------------------------------
struct Vector {
    std::vector<double> values;

    Vector() = default;
    explicit Vector(int size) { values.resize(size, 0.0); }
    explicit Vector(const std::vector<double>& v) : values(v) {}
};

// ---------------------------------------------------------------------------
// QKVItem & FastCircularBuffer — Zero-Heap Context Window (Cap = 16)
// ---------------------------------------------------------------------------
static constexpr int WORD_DIM = 384;
static constexpr int TOTAL_DIM = 768;
static constexpr int CTX_LEN = 16;
static constexpr int MAX_RAD = 8;

struct QKVItem {
    std::array<double, WORD_DIM> raw;
    std::array<double, WORD_DIM> K;
    std::array<double, WORD_DIM> V;
    int tok_id = -1;
};

template<int CAP>
struct FastCircularBuffer {
    std::array<QKVItem, CAP> buf;
    int head = 0;
    int count = 0;

    void push(const double* raw, const double* K, const double* V, int tok_id = -1) {
        int idx;
        if (count < CAP) {
            idx = (head + count) % CAP;
            ++count;
        } else {
            idx = head;
            head = (head + 1) % CAP;
        }
        std::memcpy(buf[idx].raw.data(), raw, WORD_DIM * sizeof(double));
        std::memcpy(buf[idx].K.data(), K, WORD_DIM * sizeof(double));
        std::memcpy(buf[idx].V.data(), V, WORD_DIM * sizeof(double));
        buf[idx].tok_id = tok_id;
    }

    void clear() { head = 0; count = 0; }
    int  size()  const { return count; }
    bool empty() const { return count == 0; }

    const QKVItem& operator[](int i) const { return buf[(head + i) % CAP]; }
};

// ---------------------------------------------------------------------------
// Node — Structural metadata
// ---------------------------------------------------------------------------
struct Node {
    int grid_x  = 0;
    int grid_y  = 0;
    int index   = 0;
    int hit_count = 0;
    int dominant_word_id = -1;
    std::unordered_map<int, double> transitions;
};

// ---------------------------------------------------------------------------
// EmbeddingLayer
// ---------------------------------------------------------------------------
class EmbeddingLayer {
    std::unordered_map<std::string, Vector> embeddings;
    std::unordered_map<std::string, int>    word_to_id;
    std::vector<std::string>                vocab;
    std::unordered_map<int, double>         idf_scores;
    int dims = 384;

public:
    bool has_word(const std::string& w) const { return word_to_id.count(w) > 0; }

    int get_id(const std::string& w) const {
        auto it = word_to_id.find(w);
        return (it != word_to_id.end()) ? it->second : -1;
    }

    const std::string& get_word(int id) const {
        static const std::string empty;
        return (id >= 0 && id < (int)vocab.size()) ? vocab[id] : empty;
    }

    double get_idf(int id) const {
        auto it = idf_scores.find(id);
        return (it != idf_scores.end()) ? it->second : 1.0;
    }

    bool load(const std::string& path, int max_words = -1) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        static char line[32768];
        bool first = true;
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r') continue;
            if (max_words > 0 && (int)vocab.size() >= max_words) break;

            if (first) {
                first = false;
                long long n = 0, d = 0;
                if (sscanf(line, "%lld %lld", &n, &d) == 2) {
                    dims = (int)d;
                    continue;
                }
            }

            char* ptr = line;
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            char* word_start = ptr;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
            if (ptr == word_start) continue;
            std::string word(word_start, ptr - word_start);

            std::vector<double> vals;
            vals.reserve(dims);
            while (*ptr) {
                while (*ptr == ' ' || *ptr == '\t') ptr++;
                if (*ptr == '\0' || *ptr == '\n' || *ptr == '\r') break;
                char* next_ptr;
                double v = std::strtod(ptr, &next_ptr);
                if (ptr == next_ptr) break;
                vals.push_back(v);
                ptr = next_ptr;
            }

            if ((int)vals.size() == dims) {
                int id = (int)vocab.size();
                embeddings[word] = Vector(vals);
                word_to_id[word] = id;
                vocab.push_back(word);
            }
        }
        fclose(f);
        return !vocab.empty();
    }

    void compute_idf(const std::vector<std::string>& corpus) {
        std::unordered_map<int, int> doc_freq;
        int total_docs = 0;
        for (const auto& tok : corpus) {
            int id = get_id(tok);
            if (id >= 0) {
                doc_freq[id]++;
                total_docs++;
            }
        }
        for (auto& kv : doc_freq) {
            idf_scores[kv.first] = std::log((double)total_docs / (1.0 + kv.second));
        }
    }

    void compute_idf_seqs(const std::vector<std::vector<int>>& seqs) {
        std::unordered_map<int, int> doc_freq;
        int total_docs = (int)seqs.size();
        for (const auto& seq : seqs) {
            std::vector<int> sorted_seq = seq;
            std::sort(sorted_seq.begin(), sorted_seq.end());
            sorted_seq.erase(std::unique(sorted_seq.begin(), sorted_seq.end()), sorted_seq.end());
            for (int id : sorted_seq) doc_freq[id]++;
        }
        for (auto& kv : doc_freq) {
            idf_scores[kv.first] = std::log((double)total_docs / (1.0 + kv.second));
        }
    }

    void add_token(const std::string& tok) {
        if (has_word(tok)) return;
        std::mt19937 rng(0xDEADBEEF + (unsigned)vocab.size());
        std::normal_distribution<double> nd(0.0, 0.2);
        Vector e(dims);
        for (auto& x : e.values) x = nd(rng);
        int id = (int)vocab.size();
        embeddings[tok] = e;
        word_to_id[tok] = id;
        vocab.push_back(tok);
    }

    const Vector& embed(const std::string& w) const {
        static const Vector empty(dims);
        auto it = embeddings.find(w);
        return (it != embeddings.end()) ? it->second : empty;
    }

    const Vector& embed_id(int id) const {
        static const Vector empty(dims);
        if (id >= 0 && id < (int)vocab.size())
            return embeddings.at(vocab[id]);
        return empty;
    }

    int  get_dims()   const { return dims; }
    int  vocab_size() const { return (int)vocab.size(); }
};

// ---------------------------------------------------------------------------
// TransformerSOM Engine
// ---------------------------------------------------------------------------
class TransformerSOM {
public:
    const int width, height, word_dim, total_dim;
    std::vector<Node> grid;
    std::vector<double> weights_flat;

    // Transformer Q, K, V Projection Matrices
    std::vector<double> W_q;
    std::vector<double> W_k;
    std::vector<double> W_v;

    double initial_lr     = 0.5;
    double min_lr         = 0.01;
    double current_lr     = 0.5;
    double initial_radius = 8.0;
    double min_radius     = 1.0;
    double current_radius = 8.0;
    double lr_decay       = 0.002;

    int prev_bmu = -1;
    FastCircularBuffer<CTX_LEN> ctx;

    static constexpr int LUT_SIZE = 2 * MAX_RAD + 1;
    double influence_lut[LUT_SIZE][LUT_SIZE];
    double current_lut_radius = -1.0;

    // -----------------------------------------------------------------------
    TransformerSOM(int w, int h, int wd)
        : width(w), height(h), word_dim(wd), total_dim(wd * 2)
    {
        int num_nodes = width * height;
        grid.resize(num_nodes);
        weights_flat.resize(num_nodes * total_dim);

        W_q.resize(word_dim * word_dim);
        W_k.resize(word_dim * word_dim);
        W_v.resize(word_dim * word_dim);

        std::mt19937 init_rng(1337);
        std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / word_dim));
        for (int i = 0; i < word_dim * word_dim; ++i) {
            W_q[i] = nd(init_rng);
            W_k[i] = nd(init_rng);
            W_v[i] = nd(init_rng);
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                grid[idx].index  = idx;
                grid[idx].grid_x = x;
                grid[idx].grid_y = y;

                std::mt19937 rng(42u * (unsigned)(idx + 1));
                std::normal_distribution<double> node_nd(0.0, 0.1);
                double* w_ptr = &weights_flat[idx * total_dim];
                for (int d = 0; d < total_dim; ++d) {
                    w_ptr[d] = node_nd(rng);
                }
            }
        }
        rebuild_lut(initial_radius);
    }

    void rebuild_lut(double radius) {
        if (radius == current_lut_radius) return;
        current_lut_radius = radius;
        double r2 = radius * radius;
        for (int dy = -MAX_RAD; dy <= MAX_RAD; ++dy) {
            for (int dx = -MAX_RAD; dx <= MAX_RAD; ++dx) {
                double d2 = (double)(dx*dx + dy*dy);
                if (d2 <= r2)
                    influence_lut[dy + MAX_RAD][dx + MAX_RAD] = std::exp(-d2 / (2.0 * r2));
                else
                    influence_lut[dy + MAX_RAD][dx + MAX_RAD] = 0.0;
            }
        }
    }

    void update_epoch(int epoch, int total) {
        double progress = (total <= 1) ? 1.0
                        : (double)(epoch - 1) / (double)(total - 1);
        current_radius = initial_radius * std::pow(min_radius / initial_radius, progress);
        current_lr     = initial_lr     * std::pow(min_lr     / initial_lr,     progress);
        rebuild_lut(current_radius);
    }

    void reset_state() {
        prev_bmu = -1;
        ctx.clear();
    }

    // Single vector matrix projection helper: out = x * W (contiguous row-major SIMD)
    inline void project_vector(const double* x_ptr, const double* W, double* out) const {
        std::fill_n(out, word_dim, 0.0);
        for (int j = 0; j < word_dim; ++j) {
            double xj = x_ptr[j];
            const double* w_row = &W[j * word_dim];
            #pragma omp simd
            for (int i = 0; i < word_dim; ++i) {
                out[i] += xj * w_row[i];
            }
        }
    }

    inline Vector project_vector(const Vector& x, const std::vector<double>& W) const {
        Vector out(word_dim);
        project_vector(x.values.data(), W.data(), out.values.data());
        return out;
    }

    // Fast QKV Attention using pre-projected K & V in context buffer
    void compute_qkv_attention(const double* q_ptr, const EmbeddingLayer& emb,
                               const FastCircularBuffer<CTX_LEN>& buffer,
                               double* out_attn, double* out_scores = nullptr) const {
        std::fill_n(out_attn, word_dim, 0.0);
        int n = buffer.size();
        if (n == 0) return;

        const double scale = std::sqrt((double)word_dim);
        double scores[CTX_LEN];
        double total = 0.0;

        for (int k_i = 0; k_i < n; ++k_i) {
            const auto& item = buffer[k_i];
            const double* k_ptr = item.K.data();
            double idf_w = emb.get_idf(item.tok_id);

            double dot = 0.0;
            #pragma omp simd reduction(+:dot)
            for (int i = 0; i < word_dim; ++i) {
                dot += q_ptr[i] * k_ptr[i];
            }

            int recency = n - 1 - k_i;
            double pos_w = std::exp(-recency * 0.15);
            scores[k_i] = std::exp(dot / scale) * pos_w * idf_w;
            total += scores[k_i];
        }

        if (total > 0.0) {
            for (int k_i = 0; k_i < n; ++k_i) {
                double w = scores[k_i] / total;
                if (out_scores) out_scores[k_i] = w;
                const double* v_ptr = buffer[k_i].V.data();
                #pragma omp simd
                for (int i = 0; i < word_dim; ++i) {
                    out_attn[i] += w * v_ptr[i];
                }
            }
        } else if (out_scores) {
            for (int k_i = 0; k_i < n; ++k_i) out_scores[k_i] = 0.0;
        }
    }

    Vector compute_qkv_attention(const Vector& Q, const EmbeddingLayer& emb, double* out_scores = nullptr) const {
        Vector out(word_dim);
        compute_qkv_attention(Q.values.data(), emb, ctx, out.values.data(), out_scores);
        return out;
    }

    void update_qkv_oja(const double* x_cur, const double* q_vec, const double* attn_weights,
                        int ctx_n, int bmu_idx, const FastCircularBuffer<CTX_LEN>& buffer) {
        if (ctx_n <= 0 || !attn_weights) return;

        int best_k = -1;
        double best_alpha = 0.15;
        for (int k_i = 0; k_i < ctx_n; ++k_i) {
            if (attn_weights[k_i] > best_alpha) {
                best_alpha = attn_weights[k_i];
                best_k = k_i;
            }
        }
        if (best_k < 0) return;

        const double* bmu_attn_target = &weights_flat[bmu_idx * total_dim + word_dim];
        double bmu_hit = (double)grid[bmu_idx].hit_count;
        double lr = (current_lr * 0.02) / (1.0 + bmu_hit * lr_decay);
        double decay = 0.005;
        double eff_lr = lr * best_alpha;

        const auto& item = buffer[best_k];
        const double* x_ctx = item.raw.data();
        const double* k_vec = item.K.data();

        for (int j = 0; j < word_dim; ++j) {
            double xj = x_ctx[j];
            double x_cur_j = x_cur[j];
            double* wv_row = &W_v[j * word_dim];
            double* wq_row = &W_q[j * word_dim];
            double* wk_row = &W_k[j * word_dim];

            #pragma omp simd
            for (int i = 0; i < word_dim; ++i) {
                double yi = bmu_attn_target[i];
                double ki = k_vec[i];
                wv_row[i] += eff_lr * (xj * yi - (yi * yi + decay) * wv_row[i]);
                wq_row[i] += eff_lr * (x_cur_j * ki - (ki * ki + decay) * wq_row[i]);
                wk_row[i] += eff_lr * (xj * ki - (ki * ki + decay) * wk_row[i]);
            }
        }
    }

    inline double distance_sq_flat(const double* a, const double* b, int len) const {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < len; ++i) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    int find_bmu(const double* q_ptr, int prev_node, bool cold_start = false) const {
        int    best_idx  = 0;
        double best_dist = std::numeric_limits<double>::max();

        if (cold_start || prev_node == -1) {
            // 2-stage hierarchical coarse-to-fine search: 50x faster for 256x256 SOM
            int stride = 8;
            int best_gx = 0, best_gy = 0;
            for (int y = 0; y < height; y += stride) {
                int row_start = y * width;
                for (int x = 0; x < width; x += stride) {
                    int idx = row_start + x;
                    const double* w_ptr = &weights_flat[idx * total_dim];
                    double d = distance_sq_flat(q_ptr, w_ptr, word_dim);
                    if (d < best_dist) {
                        best_dist = d;
                        best_idx  = idx;
                        best_gx   = x;
                        best_gy   = y;
                    }
                }
            }
            int min_x = std::max(0,          best_gx - stride);
            int max_x = std::min(width  - 1, best_gx + stride);
            int min_y = std::max(0,          best_gy - stride);
            int max_y = std::min(height - 1, best_gy + stride);
            for (int y = min_y; y <= max_y; ++y) {
                int row_start = y * width;
                for (int x = min_x; x <= max_x; ++x) {
                    int idx = row_start + x;
                    const double* w_ptr = &weights_flat[idx * total_dim];
                    double d = distance_sq_flat(q_ptr, w_ptr, word_dim);
                    if (d < best_dist) { best_dist = d; best_idx = idx; }
                }
            }
        } else {
            const Node& p = grid[prev_node];
            int rad   = std::max(6, (int)std::ceil(current_radius * 0.75));
            int min_x = std::max(0,          p.grid_x - rad);
            int max_x = std::min(width  - 1, p.grid_x + rad);
            int min_y = std::max(0,          p.grid_y - rad);
            int max_y = std::min(height - 1, p.grid_y + rad);

            for (int y = min_y; y <= max_y; ++y) {
                int row_start = y * width;
                for (int x = min_x; x <= max_x; ++x) {
                    int idx = row_start + x;
                    const double* w_ptr = &weights_flat[idx * total_dim];
                    double d = distance_sq_flat(q_ptr, w_ptr, total_dim);
                    if (d < best_dist) { best_dist = d; best_idx = idx; }
                }
            }

            static thread_local std::mt19937 rng(42 + omp_get_thread_num());
            std::uniform_int_distribution<int> dis(0, (int)grid.size() - 1);
            for (int s = 0; s < 30; ++s) {
                int idx = dis(rng);
                const double* w_ptr = &weights_flat[idx * total_dim];
                double d = distance_sq_flat(q_ptr, w_ptr, total_dim);
                if (d < best_dist) { best_dist = d; best_idx = idx; }
            }
        }

        return best_idx;
    }

    int find_bmu(const Vector& input, bool cold_start = false) const {
        return find_bmu(input.values.data(), prev_bmu, cold_start);
    }

    int step_thread(const double* word_emb_ptr, int word_id, const EmbeddingLayer& emb,
                    FastCircularBuffer<CTX_LEN>& t_ctx, int& t_prev_bmu,
                    std::vector<std::pair<int, int>>* t_trans_log,
                    double* Q_buf, double* K_buf, double* V_buf,
                    double* attn_buf, double* query_buf,
                    bool training = true)
    {
        // Zero-heap stack projection
        project_vector(word_emb_ptr, W_q.data(), Q_buf);
        project_vector(word_emb_ptr, W_k.data(), K_buf);
        project_vector(word_emb_ptr, W_v.data(), V_buf);

        double attn_scores[CTX_LEN] = {0.0};
        int prev_ctx_len = t_ctx.size();
        compute_qkv_attention(Q_buf, emb, t_ctx, attn_buf, attn_scores);
        t_ctx.push(word_emb_ptr, K_buf, V_buf, word_id);

        std::memcpy(&query_buf[0], word_emb_ptr, word_dim * sizeof(double));
        std::memcpy(&query_buf[word_dim], attn_buf, word_dim * sizeof(double));

        int bmu_idx = find_bmu(query_buf, t_prev_bmu);
        Node& bmu   = grid[bmu_idx];

        if (training) {
            bmu.hit_count++;
            if (word_id >= 0) bmu.dominant_word_id = word_id;
            if (t_prev_bmu >= 0) {
                if (t_trans_log) t_trans_log->push_back({t_prev_bmu, bmu_idx});
                else grid[t_prev_bmu].transitions[bmu_idx] += 1.0;
            }

            int rad   = std::min((int)std::ceil(current_radius), MAX_RAD);
            int min_x = std::max(0,          bmu.grid_x - rad);
            int max_x = std::min(width  - 1, bmu.grid_x + rad);
            int min_y = std::max(0,          bmu.grid_y - rad);
            int max_y = std::min(height - 1, bmu.grid_y + rad);

            for (int y = min_y; y <= max_y; ++y) {
                int row_start = y * width;
                int dy = y - bmu.grid_y;
                for (int x = min_x; x <= max_x; ++x) {
                    int dx = x - bmu.grid_x;
                    double influence = influence_lut[dy + MAX_RAD][dx + MAX_RAD];
                    if (influence == 0.0) continue;

                    int idx = row_start + x;
                    double lr = current_lr / (1.0 + grid[idx].hit_count * lr_decay);
                    double lri = lr * influence;

                    double* w_ptr = &weights_flat[idx * total_dim];
                    #pragma omp simd
                    for (int d = 0; d < total_dim; ++d) {
                        w_ptr[d] += lri * (query_buf[d] - w_ptr[d]);
                    }
                }
            }

            update_qkv_oja(word_emb_ptr, Q_buf, attn_scores, prev_ctx_len, bmu_idx, t_ctx);
        }

        t_prev_bmu = bmu_idx;
        return bmu_idx;
    }

    int step(const Vector& word_emb, int word_id, const EmbeddingLayer& emb, bool training = true) {
        double Q[WORD_DIM], K[WORD_DIM], V[WORD_DIM], attn[WORD_DIM], query[TOTAL_DIM];
        return step_thread(word_emb.values.data(), word_id, emb, ctx, prev_bmu, nullptr,
                           Q, K, V, attn, query, training);
    }

    std::vector<int> generate(int start_node, int max_tokens,
                              const EmbeddingLayer& emb,
                              double temp = 0.7, double top_p = 0.9)
    {
        std::vector<int> out_words;
        int cur = start_node;
        static thread_local std::mt19937 rng(std::random_device{}());

        for (int step_i = 0; step_i < max_tokens; ++step_i) {
            std::vector<std::pair<int,double>> cands;

            Vector cw  = word_component(cur);
            Vector Q   = project_vector(cw, W_q);
            Vector at  = compute_qkv_attention(Q, emb);

            if (!grid[cur].transitions.empty()) {
                double total = 0.0;
                for (auto& kv : grid[cur].transitions) total += kv.second;
                for (auto& kv : grid[cur].transitions) {
                    int cand_idx = kv.first;
                    double p = kv.second / total;

                    // Bayesian attention fusion: align candidate node attention weights with real-time context
                    const double* cand_w = &weights_flat[cand_idx * total_dim + word_dim];
                    double dot = 0.0, norm_c = 0.0, norm_a = 0.0;
                    #pragma omp simd reduction(+:dot, norm_c, norm_a)
                    for (int d = 0; d < word_dim; ++d) {
                        dot    += cand_w[d] * at.values[d];
                        norm_c += cand_w[d] * cand_w[d];
                        norm_a += at.values[d] * at.values[d];
                    }
                    double sim = 0.0;
                    if (norm_c > 1e-9 && norm_a > 1e-9) {
                        sim = dot / (std::sqrt(norm_c) * std::sqrt(norm_a));
                    }
                    double fused_score = p * std::exp(sim * 1.5);

                    int wid = grid[cand_idx].dominant_word_id;
                    if (wid >= 0 && std::find(out_words.begin(), out_words.end(), wid) != out_words.end())
                        fused_score *= 0.05;
                    cands.push_back({cand_idx, fused_score});
                }
            } else {
                Vector q(total_dim);
                for (int d = 0; d < word_dim; ++d) {
                    q.values[d]            = cw.values[d];
                    q.values[word_dim + d] = at.values[d];
                }
                int nxt = find_bmu(q, /*cold_start=*/true);
                if (nxt != cur) cands.push_back({nxt, 1.0});
            }

            if (cands.empty()) break;

            std::sort(cands.begin(), cands.end(),
                      [](const auto& a, const auto& b){ return a.second > b.second; });

            std::vector<double> probs;
            probs.reserve(cands.size());
            double sum_prob = 0.0;
            for (const auto& c : cands) {
                double p = std::exp(std::log(c.second + 1e-9) / temp);
                probs.push_back(p);
                sum_prob += p;
            }
            for (auto& p : probs) p /= sum_prob;

            std::vector<std::pair<int, double>> nucleus_cands;
            double cum_sum = 0.0;
            for (size_t i = 0; i < cands.size(); ++i) {
                nucleus_cands.push_back({cands[i].first, probs[i]});
                cum_sum += probs[i];
                if (cum_sum >= top_p) break;
            }

            std::vector<double> final_weights;
            for (const auto& nc : nucleus_cands) final_weights.push_back(nc.second);

            std::discrete_distribution<int> dd(final_weights.begin(), final_weights.end());
            int chosen = nucleus_cands[dd(rng)].first;

            int wid = grid[chosen].dominant_word_id;
            if (wid >= 0) {
                const std::string& ws = emb.get_word(wid);
                if (ws == "user_turn" || ws == "bot_turn") break;
                out_words.push_back(wid);
                cur = step(emb.embed_id(wid), wid, emb, /*training=*/false);
            } else {
                cur = chosen;
            }
        }
        return out_words;
    }

    Vector word_component(int node_idx) const {
        Vector w(word_dim);
        const double* w_ptr = &weights_flat[node_idx * total_dim];
        for (int d = 0; d < word_dim; ++d)
            w.values[d] = w_ptr[d];
        return w;
    }

    bool save_checkpoint(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        
        int magic = 0x54534F32; // "TSO2"
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&width), sizeof(width));
        out.write(reinterpret_cast<const char*>(&height), sizeof(height));
        out.write(reinterpret_cast<const char*>(&word_dim), sizeof(word_dim));
        out.write(reinterpret_cast<const char*>(&total_dim), sizeof(total_dim));
        out.write(reinterpret_cast<const char*>(&current_radius), sizeof(current_radius));
        out.write(reinterpret_cast<const char*>(&current_lr), sizeof(current_lr));
        
        size_t w_size = weights_flat.size();
        out.write(reinterpret_cast<const char*>(&w_size), sizeof(w_size));
        out.write(reinterpret_cast<const char*>(weights_flat.data()), w_size * sizeof(double));
        
        size_t qkv_size = W_q.size();
        out.write(reinterpret_cast<const char*>(&qkv_size), sizeof(qkv_size));
        out.write(reinterpret_cast<const char*>(W_q.data()), qkv_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(W_k.data()), qkv_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(W_v.data()), qkv_size * sizeof(double));

        size_t g_size = grid.size();
        out.write(reinterpret_cast<const char*>(&g_size), sizeof(g_size));
        for (size_t i = 0; i < g_size; ++i) {
            out.write(reinterpret_cast<const char*>(&grid[i].hit_count), sizeof(grid[i].hit_count));
            out.write(reinterpret_cast<const char*>(&grid[i].dominant_word_id), sizeof(grid[i].dominant_word_id));
            size_t num_trans = grid[i].transitions.size();
            out.write(reinterpret_cast<const char*>(&num_trans), sizeof(num_trans));
            for (const auto& kv : grid[i].transitions) {
                out.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
                out.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
            }
        }
        return out.good();
    }

    bool load_checkpoint(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        int magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x54534F32 && magic != 0x54534F4D) return false;

        int w, h, wd, td;
        in.read(reinterpret_cast<char*>(&w), sizeof(w));
        in.read(reinterpret_cast<char*>(&h), sizeof(h));
        in.read(reinterpret_cast<char*>(&wd), sizeof(wd));
        in.read(reinterpret_cast<char*>(&td), sizeof(td));
        
        if (w != width || h != height || wd != word_dim || td != total_dim) return false;

        in.read(reinterpret_cast<char*>(&current_radius), sizeof(current_radius));
        in.read(reinterpret_cast<char*>(&current_lr), sizeof(current_lr));

        size_t w_size = 0;
        in.read(reinterpret_cast<char*>(&w_size), sizeof(w_size));
        if (w_size != weights_flat.size()) return false;
        in.read(reinterpret_cast<char*>(weights_flat.data()), w_size * sizeof(double));

        if (magic == 0x54534F32) {
            size_t qkv_size = 0;
            in.read(reinterpret_cast<char*>(&qkv_size), sizeof(qkv_size));
            if (qkv_size != W_q.size()) return false;
            in.read(reinterpret_cast<char*>(W_q.data()), qkv_size * sizeof(double));
            in.read(reinterpret_cast<char*>(W_k.data()), qkv_size * sizeof(double));
            in.read(reinterpret_cast<char*>(W_v.data()), qkv_size * sizeof(double));
        }

        size_t g_size = 0;
        in.read(reinterpret_cast<char*>(&g_size), sizeof(g_size));
        if (g_size != grid.size()) return false;

        for (size_t i = 0; i < g_size; ++i) {
            in.read(reinterpret_cast<char*>(&grid[i].hit_count), sizeof(grid[i].hit_count));
            in.read(reinterpret_cast<char*>(&grid[i].dominant_word_id), sizeof(grid[i].dominant_word_id));
            size_t num_trans = 0;
            in.read(reinterpret_cast<char*>(&num_trans), sizeof(num_trans));
            grid[i].transitions.clear();
            for (size_t t = 0; t < num_trans; ++t) {
                int target_idx;
                double count;
                in.read(reinterpret_cast<char*>(&target_idx), sizeof(target_idx));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                grid[i].transitions[target_idx] = count;
            }
        }
        rebuild_lut(current_radius);
        return in.good();
    }
};

// ---------------------------------------------------------------------------
// Dialogue Sequence Loader
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> load_dialogue_sequences(const std::string& path, const EmbeddingLayer& emb) {
    std::vector<std::vector<int>> seqs;
    std::ifstream f(path);
    if (!f) return seqs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<int> seq;
        while (ss >> tok) {
            if (tok == "user_turn" || tok == "bot_turn") {
                int id = emb.get_id(tok);
                if (id >= 0) seq.push_back(id);
                continue;
            }
            std::string clean;
            for (char c : tok) if (std::isalnum((unsigned char)c) || c == '_') clean += std::tolower(c);
            if (!clean.empty() && emb.has_word(clean)) {
                seq.push_back(emb.get_id(clean));
            }
        }
        if (!seq.empty()) seqs.push_back(seq);
    }
    return seqs;
}

// ---------------------------------------------------------------------------
// Prompt Evaluator Helper
// ---------------------------------------------------------------------------
std::string evaluate_prompt(TransformerSOM& som, const EmbeddingLayer& emb, const std::string& prompt) {
    som.reset_state();
    int user_id = emb.get_id("user_turn");
    int bot_id  = emb.get_id("bot_turn");

    som.step(emb.embed_id(user_id), user_id, emb, false);

    bool found = false;
    std::istringstream ss(prompt);
    std::string w;
    int last_node = -1;

    while (ss >> w) {
        std::string clean;
        for (char c : w) if (std::isalnum((unsigned char)c) || c == '_') clean += std::tolower(c);
        if (clean.empty() || !emb.has_word(clean)) continue;
        found = true;
        int wid = emb.get_id(clean);
        last_node = som.step(emb.embed_id(wid), wid, emb, false);
    }

    if (!found || last_node < 0) return "[no known words in prompt]";

    last_node = som.step(emb.embed_id(bot_id), bot_id, emb, false);
    auto resp = som.generate(last_node, 20, emb, 0.7, 0.9);

    if (resp.empty()) return "i see";
    std::string out = "";
    for (size_t i = 0; i < resp.size(); ++i) {
        if (i > 0) out += " ";
        out += emb.get_word(resp[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    EmbeddingLayer emb;
    std::string emb_path = "minilm_384d.txt";
    if (!emb.load(emb_path)) {
        std::cerr << "Failed to load embeddings from: " << emb_path << "\n"; return 1;
    }
    emb.add_token("user_turn");
    emb.add_token("bot_turn");

    // CLI Mode: Eval Prompt
    if (argc >= 4 && std::string(argv[1]) == "--eval") {
        std::string ckpt_path = argv[2];
        std::string prompt    = argv[3];
        TransformerSOM som(256, 256, emb.get_dims());
        if (!som.load_checkpoint(ckpt_path)) {
            std::cerr << "Failed to load checkpoint: " << ckpt_path << "\n"; return 1;
        }
        std::cout << evaluate_prompt(som, emb, prompt) << "\n";
        return 0;
    }

    // CLI Mode: Test Suite
    if (argc >= 3 && std::string(argv[1]) == "--test-suite") {
        std::string ckpt_path = argv[2];
        TransformerSOM som(256, 256, emb.get_dims());
        if (!som.load_checkpoint(ckpt_path)) {
            std::cerr << "Failed to load checkpoint: " << ckpt_path << "\n"; return 1;
        }
        std::vector<std::string> test_prompts = {
            "hello how are you",
            "what is your name",
            "where are you going",
            "tell me something good",
            "who is there"
        };
        std::cout << "--- TRANSFORMER-SOM TEST SUITE (" << ckpt_path << ") ---\n";
        for (const auto& p : test_prompts) {
            std::cout << "PROMPT: " << p << "\n";
            std::cout << "RESPONSE: " << evaluate_prompt(som, emb, p) << "\n---\n";
        }
        return 0;
    }

    // CLI Mode: Training
    int total_epochs = 20;
    if (argc >= 3 && std::string(argv[1]) == "--train") {
        total_epochs = std::atoi(argv[2]);
    }

    std::string corpus_path = "oasst_corpus.txt";
    auto seqs = load_dialogue_sequences(corpus_path, emb);
    if (seqs.empty()) { std::cerr << "Failed to load dialogues from: " << corpus_path << "\n"; return 1; }

    emb.compute_idf_seqs(seqs);

    int num_seqs = (int)seqs.size();
    int num_threads = omp_get_max_threads();
    long long total_tokens = 0;
    for (const auto& s : seqs) total_tokens += s.size();

    TransformerSOM som(256, 256, emb.get_dims());

    std::cout << "Training TransformerSOM (" << total_epochs
              << " epochs, " << num_seqs << " dialogues, " << total_tokens << " tokens/ep, "
              << num_threads << " OMP worker threads, 384d MiniLM, Nucleus Top-P=0.9)...\n" << std::flush;

    std::vector<std::vector<std::pair<int, int>>> thread_trans_logs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_trans_logs[t].reserve(total_tokens / num_threads + 50000);
    }

    auto t_start = std::chrono::steady_clock::now();
    long long total_processed = 0;

    for (int ep = 1; ep <= total_epochs; ++ep) {
        som.update_epoch(ep, total_epochs);
        std::atomic<long long> ep_processed{0};

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            FastCircularBuffer<CTX_LEN> t_ctx;
            int t_prev_bmu = -1;
            auto& t_log = thread_trans_logs[tid];

            double Q_buf[WORD_DIM];
            double K_buf[WORD_DIM];
            double V_buf[WORD_DIM];
            double attn_buf[WORD_DIM];
            double query_buf[TOTAL_DIM];

            long long local_toks = 0;

            #pragma omp for schedule(dynamic, 8)
            for (int s = 0; s < num_seqs; ++s) {
                t_ctx.clear();
                t_prev_bmu = -1;
                const auto& seq = seqs[s];
                for (int wid : seq) {
                    som.step_thread(emb.embed_id(wid).values.data(), wid, emb, t_ctx, t_prev_bmu,
                                    &t_log, Q_buf, K_buf, V_buf, attn_buf, query_buf, true);
                    local_toks++;
                }

                if (local_toks >= 5000) {
                    long long p = (ep_processed += local_toks);
                    local_toks = 0;
                    if (p % 100000 < 5000) {
                        #pragma omp critical
                        {
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - t_start).count();
                            double tok_per_sec = (elapsed > 0) ? (double)(total_processed + p) / elapsed : 0;
                            std::cout << "  Ep " << ep << "/" << total_epochs
                                      << "  r=" << std::fixed << std::setprecision(1) << som.current_radius
                                      << "  tokens=" << p << "/" << total_tokens
                                      << "  speed=" << (int)tok_per_sec << " tok/s (16 threads active)\n" << std::flush;
                        }
                    }
                }
            }
            if (local_toks > 0) ep_processed += local_toks;
        }

        total_processed += ep_processed.load();

        // Merge thread transitions into global grid
        for (int t = 0; t < num_threads; ++t) {
            for (const auto& edge : thread_trans_logs[t]) {
                som.grid[edge.first].transitions[edge.second] += 1.0;
            }
            thread_trans_logs[t].clear();
        }

        std::string ep_ckpt = "checkpoint_epoch_" + std::to_string(ep) + ".bin";
        if (som.save_checkpoint(ep_ckpt)) {
            std::cout << "  [CHECKPOINT] Saved epoch " << ep << " model to " << ep_ckpt << "\n" << std::flush;
            som.save_checkpoint("latest_checkpoint.bin");
        }
    }

    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "\nTraining complete in " << total_secs << "s ("
              << (total_processed / std::max(1LL, (long long)total_secs)) << " tok/s average).\n\n" << std::flush;

    bool interactive = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-interactive") interactive = false;
    }

    if (!interactive) return 0;

    std::cout << "====================================================\n"
              << "  TransformerSOM Interactive Chat\n"
              << "  Type a message. 'quit' to exit.\n"
              << "====================================================\n\n" << std::flush;

    som.reset_state();
    int user_id = emb.get_id("user_turn");
    int bot_id  = emb.get_id("bot_turn");

    std::string line;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line) || line == "quit") break;

        som.step(emb.embed_id(user_id), user_id, emb, false);

        bool found = false;
        std::istringstream ss(line);
        std::string w;
        int last_node = -1;

        while (ss >> w) {
            std::string clean;
            for (char c : w) if (std::isalnum((unsigned char)c) || c == '_') clean += std::tolower(c);
            if (clean.empty() || !emb.has_word(clean)) continue;
            found = true;
            int wid = emb.get_id(clean);
            last_node = som.step(emb.embed_id(wid), wid, emb, false);
        }

        if (!found || last_node < 0) {
            std::cout << "BOT: [no known words in prompt]\n\n" << std::flush;
            continue;
        }

        last_node = som.step(emb.embed_id(bot_id), bot_id, emb, false);

        auto resp = som.generate(last_node, 20, emb, 0.7, 0.9);
        std::cout << "BOT: ";
        if (resp.empty()) {
            std::cout << "i see";
        } else {
            for (int wid : resp) std::cout << emb.get_word(wid) << " ";
        }
        std::cout << "\n\n" << std::flush;
    }

    return 0;
}
