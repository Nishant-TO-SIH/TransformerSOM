#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <iomanip>
#include <cctype>
#include <cstring>
#include <atomic>
#include <omp.h>
#include <immintrin.h>

// ---------------------------------------------------------------------------
// Constants & Hyperparameters for Balanced Ternary Hypercube (3^11)
// ---------------------------------------------------------------------------
constexpr int WORD_DIM    = 384;
constexpr int TOTAL_DIM   = 768; // 384 (word) + 384 (context)
constexpr int CTX_LEN     = 16;
constexpr int TERNARY_DIM = 11;  // 11 dimensions in {-1, 0, +1}
constexpr int NUM_NODES   = 177147; // 3^11 = 177,147 nodes

// ---------------------------------------------------------------------------
// Vector Struct with AVX2 Support
// ---------------------------------------------------------------------------
struct Vector {
    std::vector<double> values;
    Vector() : values(WORD_DIM, 0.0) {}
    Vector(int dim) : values(dim, 0.0) {}
    Vector(int dim, double val) : values(dim, val) {}
    Vector(const std::vector<double>& v) : values(v) {}
};

// ---------------------------------------------------------------------------
// Fast Ring-Buffer for Zero-Allocation Context Window
// ---------------------------------------------------------------------------
template <int Capacity = CTX_LEN>
class FastCircularBuffer {
public:
    struct Item {
        std::array<double, WORD_DIM> raw;
        std::array<double, WORD_DIM> K;
        std::array<double, WORD_DIM> V;
        int word_id = -1;
    };

    Item buffer[Capacity];
    int head = 0;
    int count = 0;

    void clear() {
        head = 0;
        count = 0;
    }

    void push(const double* raw_ptr, const double* k_ptr, const double* v_ptr, int wid) {
        Item& it = buffer[head];
        std::memcpy(it.raw.data(), raw_ptr, WORD_DIM * sizeof(double));
        std::memcpy(it.K.data(),   k_ptr,   WORD_DIM * sizeof(double));
        std::memcpy(it.V.data(),   v_ptr,   WORD_DIM * sizeof(double));
        it.word_id = wid;

        head = (head + 1) % Capacity;
        if (count < Capacity) count++;
    }

    int size() const { return count; }

    const Item& operator[](int i) const {
        int idx = (head - count + i) % Capacity;
        if (idx < 0) idx += Capacity;
        return buffer[idx];
    }
};

// ---------------------------------------------------------------------------
// Ternary Node Struct
// ---------------------------------------------------------------------------
struct TernaryNode {
    uint32_t id = 0; // 0 .. 177,146
    int hit_count = 0;
    int dominant_word_id = -1;
    std::unordered_map<uint32_t, double> transitions;
};

// ---------------------------------------------------------------------------
// EmbeddingLayer
// ---------------------------------------------------------------------------
class EmbeddingLayer {
    std::unordered_map<std::string, Vector> embeddings;
    std::unordered_map<std::string, int>    word_to_id;
    std::vector<std::string>                vocab;
    std::unordered_map<int, double>         idf_scores;
    int dims = WORD_DIM;

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

    bool load(const std::string& path, int max_words = 45000) {
        FILE* f = fopen(path.c_str(), "r");
        if (!f) return false;

        char line[16384];
        while (fgets(line, sizeof(line), f)) {
            if (max_words > 0 && (int)vocab.size() >= max_words) break;

            char* ptr = line;
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n' || *ptr == '\r') continue;

            char* word_start = ptr;
            while (*ptr && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
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

    const Vector& embed_id(int id) const {
        static const Vector empty(WORD_DIM, 0.0);
        if (id >= 0 && id < (int)vocab.size())
            return embeddings.at(vocab[id]);
        return empty;
    }

    int get_dims() const { return dims; }
    int vocab_size() const { return (int)vocab.size(); }
};

// ---------------------------------------------------------------------------
// TernarySOM Engine: 11-Dimensional Balanced Ternary Lattice {-1, 0, +1}^11
// Total nodes: 3^11 = 177,147
// ---------------------------------------------------------------------------
class TernarySOM {
public:
    const int word_dim = WORD_DIM;
    const int total_dim = TOTAL_DIM;
    const int num_nodes = NUM_NODES; // 177,147

    std::vector<TernaryNode> grid;
    std::vector<double> weights_flat;

    // Transformer Q, K, V Projection Matrices
    std::vector<double> W_q;
    std::vector<double> W_k;
    std::vector<double> W_v;

    // Ternary Lattice Geometry
    uint32_t pow3[TERNARY_DIM]; // 3^0 .. 3^10
    std::vector<int8_t> node_trits; // NUM_NODES * TERNARY_DIM (in {-1, 0, +1})
    std::vector<uint32_t> neighbors_d1_flat; // NUM_NODES * 22
    std::vector<uint8_t> num_neighbors_d1;   // NUM_NODES

    // Neighborhood Radius in Manhattan/Trit distance (max 3 steps)
    double initial_radius = 3.0;
    double min_radius     = 1.0;
    double current_radius = 3.0;

    double initial_lr     = 0.5;
    double min_lr         = 0.05;
    double current_lr     = 0.5;
    double lr_decay       = 0.002;

    int prev_bmu = -1;
    FastCircularBuffer<CTX_LEN> ctx;

    // Manhattan Influence weights [0..3]
    double manhattan_influence[4] = {1.0, 0.6, 0.2, 0.05};

    TernarySOM() {
        grid.resize(num_nodes);
        weights_flat.resize((size_t)num_nodes * total_dim);

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

        // 1. Precompute Powers of 3: 3^0 to 3^10
        uint32_t p = 1;
        for (int d = 0; d < TERNARY_DIM; ++d) {
            pow3[d] = p;
            p *= 3;
        }

        // 2. Precompute Trits and 1-Step Neighbors for all 177,147 nodes
        node_trits.resize((size_t)num_nodes * TERNARY_DIM);
        neighbors_d1_flat.resize((size_t)num_nodes * 22);
        num_neighbors_d1.resize(num_nodes);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_nodes; ++i) {
            grid[i].id = (uint32_t)i;
            int temp = i;
            int8_t* trits = &node_trits[(size_t)i * TERNARY_DIM];

            for (int d = 0; d < TERNARY_DIM; ++d) {
                int r = temp % 3; // 0, 1, 2
                trits[d] = (int8_t)(r - 1); // -1, 0, +1
                temp /= 3;
            }

            // Find valid 1-step neighbors along 11 axes
            uint32_t* nbr_ptr = &neighbors_d1_flat[(size_t)i * 22];
            uint8_t count = 0;
            for (int d = 0; d < TERNARY_DIM; ++d) {
                int t = trits[d];
                if (t < 1) { // can step +1 (towards +1)
                    nbr_ptr[count++] = i + pow3[d];
                }
                if (t > -1) { // can step -1 (towards -1)
                    nbr_ptr[count++] = i - pow3[d];
                }
            }
            num_neighbors_d1[i] = count;
        }

        rebuild_lut(initial_radius);
    }

    void init_embedding_manifold(const EmbeddingLayer& emb) {
        int v_size = emb.vocab_size();
        if (v_size == 0) return;

        std::normal_distribution<double> jitter(0.0, 0.05);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_nodes; ++i) {
            std::mt19937 node_rng(1337u + (unsigned)i * 37u);
            int wid1 = node_rng() % v_size;
            int wid2 = node_rng() % v_size;
            grid[i].dominant_word_id = wid1;

            const auto& e1 = emb.embed_id(wid1);
            const auto& e2 = emb.embed_id(wid2);
            double* w_ptr = &weights_flat[(size_t)i * total_dim];

            double norm1 = 0.0, norm2 = 0.0;
            for (int d = 0; d < word_dim; ++d) {
                w_ptr[d] = e1.values[d] + jitter(node_rng);
                norm1 += w_ptr[d] * w_ptr[d];

                w_ptr[word_dim + d] = e2.values[d] + jitter(node_rng);
                norm2 += w_ptr[word_dim + d] * w_ptr[word_dim + d];
            }
            norm1 = std::sqrt(norm1);
            norm2 = std::sqrt(norm2);
            if (norm1 > 1e-6) {
                for (int d = 0; d < word_dim; ++d) w_ptr[d] /= norm1;
            }
            if (norm2 > 1e-6) {
                for (int d = 0; d < word_dim; ++d) w_ptr[word_dim + d] /= norm2;
            }
        }
    }

    void rebuild_lut(double radius) {
        double r2 = radius * radius;
        for (int d = 0; d <= 3; ++d) {
            if ((double)d <= radius) {
                manhattan_influence[d] = std::exp(-(double)(d * d) / (2.0 * r2));
            } else {
                manhattan_influence[d] = 0.0;
            }
        }
        manhattan_influence[0] = 1.0;
    }

    void update_epoch(int epoch, int total) {
        double progress = (total <= 1) ? 0.0 : (double)(epoch - 1) / (double)(total - 1);
        current_radius = initial_radius * std::pow(min_radius / initial_radius, progress);
        current_lr     = initial_lr     * std::pow(min_lr     / initial_lr,     progress);
        rebuild_lut(current_radius);
    }

    void reset_state() {
        prev_bmu = -1;
        ctx.clear();
    }

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

    Vector project_vector(const Vector& x, const std::vector<double>& W) const {
        Vector out(word_dim, 0.0);
        project_vector(x.values.data(), W.data(), out.values.data());
        return out;
    }

    void compute_qkv_attention(const double* Q_buf,
                               const EmbeddingLayer& emb,
                               const FastCircularBuffer<CTX_LEN>& buffer,
                               double* attn_out,
                               double* attn_scores_out) const
    {
        int ctx_len = buffer.size();
        if (ctx_len == 0) {
            std::fill_n(attn_out, word_dim, 0.0);
            return;
        }

        double inv_scale = 1.0 / std::sqrt((double)word_dim);
        double max_score = -1e9;

        for (int i = 0; i < ctx_len; ++i) {
            const auto& item = buffer[i];
            const double* k_ptr = item.K.data();

            double dot = 0.0;
            #pragma omp simd reduction(+:dot)
            for (int d = 0; d < word_dim; ++d) {
                dot += Q_buf[d] * k_ptr[d];
            }
            double idf = (item.word_id >= 0) ? emb.get_idf(item.word_id) : 1.0;
            double score = (dot * inv_scale) * idf;
            attn_scores_out[i] = score;
            if (score > max_score) max_score = score;
        }

        double sum_exp = 0.0;
        for (int i = 0; i < ctx_len; ++i) {
            double e = std::exp(attn_scores_out[i] - max_score);
            attn_scores_out[i] = e;
            sum_exp += e;
        }
        double inv_sum = 1.0 / (sum_exp + 1e-9);
        for (int i = 0; i < ctx_len; ++i) attn_scores_out[i] *= inv_sum;

        std::fill_n(attn_out, word_dim, 0.0);
        for (int i = 0; i < ctx_len; ++i) {
            double alpha = attn_scores_out[i];
            const double* v_ptr = buffer[i].V.data();
            #pragma omp simd
            for (int d = 0; d < word_dim; ++d) {
                attn_out[d] += alpha * v_ptr[d];
            }
        }
    }

    Vector compute_qkv_attention(const Vector& Q, const EmbeddingLayer& emb) const {
        Vector out(word_dim, 0.0);
        double scores[CTX_LEN];
        compute_qkv_attention(Q.values.data(), emb, ctx, out.values.data(), scores);
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

        const double* bmu_attn_target = &weights_flat[(size_t)bmu_idx * total_dim + word_dim];
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

    // -----------------------------------------------------------------------
    // Balanced Ternary BMU Search:
    // 1. Cold Start: 48 random anchors + 22-axis coordinate descent (~114 evals)
    // 2. Sequential: prev_bmu + 16 random anchors + 22-axis descent (~70 evals)
    // -----------------------------------------------------------------------
    int find_bmu(const double* q_ptr, int prev_node, bool cold_start = false) const {
        static thread_local std::mt19937 rng(42 + omp_get_thread_num());
        std::uniform_int_distribution<int> dis(0, num_nodes - 1);

        int    best_idx  = 0;
        double best_dist = std::numeric_limits<double>::max();

        if (cold_start || prev_node < 0) {
            // 48 stochastic seeds across the 177,147 nodes
            for (int s = 0; s < 48; ++s) {
                int idx = dis(rng);
                const double* w_ptr = &weights_flat[(size_t)idx * total_dim];
                double d = distance_sq_flat(q_ptr, w_ptr, word_dim);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx  = idx;
                }
            }

            // 22-axis coordinate descent (3 passes)
            for (int pass = 0; pass < 3; ++pass) {
                int step_winner = best_idx;
                int n_cnt = num_neighbors_d1[best_idx];
                const uint32_t* nbrs = &neighbors_d1_flat[(size_t)best_idx * 22];

                for (int k = 0; k < n_cnt; ++k) {
                    int cand = nbrs[k];
                    const double* w_ptr = &weights_flat[(size_t)cand * total_dim];
                    double d = distance_sq_flat(q_ptr, w_ptr, word_dim);
                    if (d < best_dist) {
                        best_dist   = d;
                        step_winner = cand;
                    }
                }
                if (step_winner == best_idx) break;
                best_idx = step_winner;
            }
        } else {
            // Sequential start from prev_bmu
            best_idx  = prev_node;
            best_dist = distance_sq_flat(q_ptr, &weights_flat[(size_t)best_idx * total_dim], total_dim);

            // 16 exploration seeds
            for (int s = 0; s < 16; ++s) {
                int idx = dis(rng);
                const double* w_ptr = &weights_flat[(size_t)idx * total_dim];
                double d = distance_sq_flat(q_ptr, w_ptr, total_dim);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx  = idx;
                }
            }

            // 22-axis coordinate descent (3 passes)
            for (int pass = 0; pass < 3; ++pass) {
                int step_winner = best_idx;
                int n_cnt = num_neighbors_d1[best_idx];
                const uint32_t* nbrs = &neighbors_d1_flat[(size_t)best_idx * 22];

                for (int k = 0; k < n_cnt; ++k) {
                    int cand = nbrs[k];
                    const double* w_ptr = &weights_flat[(size_t)cand * total_dim];
                    double d = distance_sq_flat(q_ptr, w_ptr, total_dim);
                    if (d < best_dist) {
                        best_dist   = d;
                        step_winner = cand;
                    }
                }
                if (step_winner == best_idx) break;
                best_idx = step_winner;
            }
        }

        return best_idx;
    }

    int find_bmu(const Vector& input, bool cold_start = false) const {
        return find_bmu(input.values.data(), prev_bmu, cold_start);
    }

    // -----------------------------------------------------------------------
    // Balanced Ternary Step Function:
    // Neighborhood updates traverse precomputed 1-step neighbors and their outward extensions
    // -----------------------------------------------------------------------
    int step_thread(const double* word_emb_ptr, int word_id, const EmbeddingLayer& emb,
                    FastCircularBuffer<CTX_LEN>& t_ctx, int& t_prev_bmu,
                    std::vector<std::pair<uint32_t, uint32_t>>* t_trans_log,
                    double* Q_buf, double* K_buf, double* V_buf,
                    double* attn_buf, double* query_buf,
                    bool training = true)
    {
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
        TernaryNode& bmu = grid[bmu_idx];

        if (training) {
            bmu.hit_count++;
            if (word_id >= 0) bmu.dominant_word_id = word_id;
            if (t_prev_bmu >= 0) {
                if (t_trans_log) t_trans_log->push_back({(uint32_t)t_prev_bmu, (uint32_t)bmu_idx});
                else grid[t_prev_bmu].transitions[(uint32_t)bmu_idx] += 1.0;
            }

            // 1. Update BMU (Manhattan distance 0)
            double lr0 = current_lr / (1.0 + bmu.hit_count * lr_decay);
            double* w0_ptr = &weights_flat[(size_t)bmu_idx * total_dim];
            #pragma omp simd
            for (int d = 0; d < total_dim; ++d) {
                w0_ptr[d] += lr0 * (query_buf[d] - w0_ptr[d]);
            }

            // 2. Update 1-step neighbors (Manhattan distance 1)
            double w_inf1 = manhattan_influence[1];
            if (w_inf1 > 0.0) {
                int n_cnt = num_neighbors_d1[bmu_idx];
                const uint32_t* nbrs = &neighbors_d1_flat[(size_t)bmu_idx * 22];
                for (int k = 0; k < n_cnt; ++k) {
                    int idx = nbrs[k];
                    double lr = (current_lr * w_inf1) / (1.0 + grid[idx].hit_count * lr_decay);
                    double* w_ptr = &weights_flat[(size_t)idx * total_dim];
                    #pragma omp simd
                    for (int d = 0; d < total_dim; ++d) {
                        w_ptr[d] += lr * (query_buf[d] - w_ptr[d]);
                    }
                }
            }

            // 3. Update 2-step neighbors if radius >= 1.5
            double w_inf2 = manhattan_influence[2];
            if (w_inf2 > 0.0 && current_radius >= 1.5) {
                int n_cnt1 = num_neighbors_d1[bmu_idx];
                const uint32_t* nbrs1 = &neighbors_d1_flat[(size_t)bmu_idx * 22];
                for (int k1 = 0; k1 < n_cnt1; ++k1) {
                    int n1 = nbrs1[k1];
                    int n_cnt2 = num_neighbors_d1[n1];
                    const uint32_t* nbrs2 = &neighbors_d1_flat[(size_t)n1 * 22];
                    for (int k2 = 0; k2 < n_cnt2; ++k2) {
                        int idx = nbrs2[k2];
                        if (idx == bmu_idx) continue;
                        double lr = (current_lr * w_inf2) / (1.0 + grid[idx].hit_count * lr_decay);
                        double* w_ptr = &weights_flat[(size_t)idx * total_dim];
                        #pragma omp simd
                        for (int d = 0; d < total_dim; ++d) {
                            w_ptr[d] += lr * (query_buf[d] - w_ptr[d]);
                        }
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

    Vector word_component(int idx) const {
        Vector v(word_dim);
        const double* ptr = &weights_flat[(size_t)idx * total_dim];
        for (int i = 0; i < word_dim; ++i) v.values[i] = ptr[i];
        return v;
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

            // 1. Empirical observed transitions from cur
            if (!grid[cur].transitions.empty()) {
                double total = 0.0;
                for (auto& kv : grid[cur].transitions) total += kv.second;
                for (auto& kv : grid[cur].transitions) {
                    int cand_idx = kv.first;
                    double p = kv.second / total;

                    const double* cand_w = &weights_flat[(size_t)cand_idx * total_dim + word_dim];
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
                        fused_score *= 0.01;
                    cands.push_back({cand_idx, fused_score});
                }
            }

            // 2. Include all 1-step Ternary topological neighbors
            int n_cnt = num_neighbors_d1[cur];
            const uint32_t* nbrs = &neighbors_d1_flat[(size_t)cur * 22];
            for (int k = 0; k < n_cnt; ++k) {
                int cand_idx = nbrs[k];
                const double* cand_w = &weights_flat[(size_t)cand_idx * total_dim + word_dim];
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
                double fused_score = 0.02 * std::exp(sim * 1.2);
                int wid = grid[cand_idx].dominant_word_id;
                if (wid >= 0 && std::find(out_words.begin(), out_words.end(), wid) != out_words.end())
                    fused_score *= 0.01;
                cands.push_back({cand_idx, fused_score});
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
            if (wid < 0) break;
            out_words.push_back(wid);

            step(emb.embed_id(wid), wid, emb, false);
            cur = chosen;
        }

        return out_words;
    }

    bool save_checkpoint(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;

        int magic = 0x54535431; // "TST1" = TransformerSOM Ternary v1
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
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
        if (magic != 0x54535431) return false;

        int nn, wd, td;
        in.read(reinterpret_cast<char*>(&nn), sizeof(nn));
        in.read(reinterpret_cast<char*>(&wd), sizeof(wd));
        in.read(reinterpret_cast<char*>(&td), sizeof(td));

        if (nn != num_nodes || wd != word_dim || td != total_dim) return false;

        in.read(reinterpret_cast<char*>(&current_radius), sizeof(current_radius));
        in.read(reinterpret_cast<char*>(&current_lr), sizeof(current_lr));

        size_t w_size = 0;
        in.read(reinterpret_cast<char*>(&w_size), sizeof(w_size));
        if (w_size != weights_flat.size()) return false;
        in.read(reinterpret_cast<char*>(weights_flat.data()), w_size * sizeof(double));

        size_t qkv_size = 0;
        in.read(reinterpret_cast<char*>(&qkv_size), sizeof(qkv_size));
        if (qkv_size != W_q.size()) return false;
        in.read(reinterpret_cast<char*>(W_q.data()), qkv_size * sizeof(double));
        in.read(reinterpret_cast<char*>(W_k.data()), qkv_size * sizeof(double));
        in.read(reinterpret_cast<char*>(W_v.data()), qkv_size * sizeof(double));

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
                uint32_t target = 0;
                double count = 0.0;
                in.read(reinterpret_cast<char*>(&target), sizeof(target));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                grid[i].transitions[target] = count;
            }
        }
        return in.good();
    }
};

// ---------------------------------------------------------------------------
// Helpers for Dialogue Parsing and Test Suite Evaluation
// ---------------------------------------------------------------------------
std::vector<std::vector<int>> load_dialogue_sequences(const std::string& path, const EmbeddingLayer& emb, int max_seqs = 15000) {
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
        if (!seq.empty()) {
            seqs.push_back(seq);
            if (max_seqs > 0 && (int)seqs.size() >= max_seqs) break;
        }
    }
    return seqs;
}

std::string evaluate_prompt(TernarySOM& som, const EmbeddingLayer& emb, const std::string& prompt) {
    som.reset_state();
    int user_id = emb.get_id("user_turn");
    int bot_id  = emb.get_id("bot_turn");

    som.step(emb.embed_id(user_id), user_id, emb, false);

    std::istringstream ss(prompt);
    std::string w;
    int last_node = -1;
    bool found = false;

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
    std::string out;
    for (size_t i = 0; i < resp.size(); ++i) {
        out += emb.get_word(resp[i]);
        if (i + 1 < resp.size()) out += " ";
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << "====================================================\n"
              << "  TernarySOM: 11-Dimensional Balanced Ternary Engine\n"
              << "  Nodes: 177,147 ({-1,0,+1}^11) | 22-Direction Lattice\n"
              << "====================================================\n" << std::flush;

    EmbeddingLayer emb;
    std::string emb_path = "minilm_384d.txt";
    if (!emb.load(emb_path, 45000)) {
        std::cerr << "Failed to load embeddings from: " << emb_path << "\n";
        return 1;
    }
    std::cout << "Loaded " << emb.vocab_size() << " word embeddings (" << emb.get_dims() << "d).\n" << std::flush;

    emb.add_token("user_turn");
    emb.add_token("bot_turn");

    // CLI Mode: Test Suite
    if (argc >= 3 && std::string(argv[1]) == "--test-suite") {
        std::string ckpt_path = argv[2];
        TernarySOM som;
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
        std::cout << "--- TERNARY-SOM TEST SUITE (" << ckpt_path << ") ---\n";
        for (const auto& p : test_prompts) {
            std::cout << "PROMPT: " << p << "\n";
            std::cout << "RESPONSE: " << evaluate_prompt(som, emb, p) << "\n---\n";
        }
        return 0;
    }

    int total_epochs = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--train" && i + 1 < argc) {
            total_epochs = std::atoi(argv[i + 1]);
        }
    }

    std::string corpus_path = "oasst_corpus.txt";
    auto seqs = load_dialogue_sequences(corpus_path, emb, 15000);
    if (seqs.empty()) { std::cerr << "Failed to load dialogues from: " << corpus_path << "\n"; return 1; }

    emb.compute_idf_seqs(seqs);

    int num_seqs = (int)seqs.size();
    int num_threads = omp_get_max_threads();
    long long total_tokens = 0;
    for (const auto& s : seqs) total_tokens += s.size();

    TernarySOM som;
    som.init_embedding_manifold(emb);

    std::cout << "Training TernarySOM (" << total_epochs
              << " epochs, " << num_seqs << " dialogues, " << total_tokens << " tokens/ep, "
              << num_threads << " OMP threads, 11D Ternary Hypercube, 177,147 nodes)...\n" << std::flush;

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> thread_trans_logs(num_threads);
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
                            double ep_ratio = (double)p / (double)total_tokens;
                            som.current_radius = std::max(1.0, 3.0 * (1.0 - 0.67 * ep_ratio));
                            som.current_lr     = std::max(0.05, 0.5 * (1.0 - 0.9 * ep_ratio));
                            som.rebuild_lut(som.current_radius);

                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - t_start).count();
                            double tok_per_sec = (elapsed > 0) ? (double)(total_processed + p) / elapsed : 0;
                            std::cout << "  Ep " << ep << "/" << total_epochs
                                      << "  Manhattan-r=" << std::fixed << std::setprecision(1) << som.current_radius
                                      << "  lr=" << std::setprecision(3) << som.current_lr
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

        // Diffusion: ensure any unassigned nodes inherit word from 1-step neighbor
        for (int i = 0; i < som.num_nodes; ++i) {
            if (som.grid[i].dominant_word_id < 0) {
                int n_cnt = som.num_neighbors_d1[i];
                const uint32_t* nbrs = &som.neighbors_d1_flat[(size_t)i * 22];
                for (int k = 0; k < n_cnt; ++k) {
                    int neighbor = nbrs[k];
                    if (som.grid[neighbor].dominant_word_id >= 0) {
                        som.grid[i].dominant_word_id = som.grid[neighbor].dominant_word_id;
                        break;
                    }
                }
            }
        }

        std::string ep_ckpt = "ternary_epoch_" + std::to_string(ep) + ".bin";
        if (som.save_checkpoint(ep_ckpt)) {
            std::cout << "  [CHECKPOINT] Saved ternary epoch " << ep << " model to " << ep_ckpt << "\n" << std::flush;
            som.save_checkpoint("ternary_latest.bin");
        }
    }

    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "\nTernary Training complete in " << total_secs << "s ("
              << (total_processed / std::max(1LL, (long long)total_secs)) << " tok/s average).\n\n" << std::flush;

    return 0;
}
