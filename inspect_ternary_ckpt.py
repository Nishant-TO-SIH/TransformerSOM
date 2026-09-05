import struct

with open("ternary_epoch_1.bin", "rb") as f:
    magic, nn, wd, td, r, lr = struct.unpack("<Iiiidd", f.read(4 + 4 + 4 + 4 + 8 + 8))
    print(f"magic={hex(magic)}, nn={nn}, wd={wd}, td={td}, r={r:.3f}, lr={lr:.4f}")
    w_size = struct.unpack("<Q", f.read(8))[0]
    f.seek(w_size * 8, 1) # skip weights
    qkv_size = struct.unpack("<Q", f.read(8))[0]
    f.seek(qkv_size * 8 * 3, 1) # skip QKV
    g_size = struct.unpack("<Q", f.read(8))[0]
    print(f"g_size={g_size}")

    hits = 0
    assigned = 0
    words = {}
    total_trans = 0
    total_hits = 0
    for i in range(g_size):
        hit, dom, num_trans = struct.unpack("<iiQ", f.read(4 + 4 + 8))
        total_hits += hit
        f.seek(num_trans * (4 + 8), 1) # uint32_t + double = 12 bytes per transition
        total_trans += num_trans
        if hit > 0: hits += 1
        if dom >= 0:
            assigned += 1
            words[dom] = words.get(dom, 0) + 1

    print(f"nodes with hits: {hits}/{g_size} ({hits/g_size*100:.1f}%)")
    print(f"total hits across all nodes: {total_hits}")
    print(f"nodes with dominant word: {assigned}/{g_size}")
    print(f"unique dominant words: {len(words)}")
    print(f"total transitions: {total_trans}")
    top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:10]

# Read vocab
vocab = []
with open("minilm_384d.txt", "r", encoding="utf-8") as vf:
    for line in vf:
        parts = line.split()
        if parts: vocab.append(parts[0])

vocab.append("user_turn")
vocab.append("bot_turn")

print("\nTop 10 dominant words across 11D Balanced Ternary nodes:")
for wid, count in top_words:
    word = vocab[wid] if wid < len(vocab) else f"ID_{wid}"
    print(f"  '{word}' (id {wid}): {count} nodes ({count/g_size*100:.2f}%)")
