//! Byte-pair-encoding tokenizer (GPT-2 / tiktoken style) built from the
//! model's vocabulary alone: in this vocabulary a token's id *is* its merge
//! rank, so the lowest-id mergeable pair is always merged first.
const std = @import("std");
const Model = @import("model.zig").Model;
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    ranks: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn init(gpa: Allocator, model: *const Model) !Tokenizer {
        var t: Tokenizer = .{};
        errdefer t.ranks.deinit(gpa);
        for (model.vocab, 0..) |bytes, i| {
            if (i >= model.tokens.eot or bytes.len == 0) continue;
            const gop = try t.ranks.getOrPut(gpa, bytes);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }
        return t;
    }

    pub fn deinit(t: *Tokenizer, gpa: Allocator) void {
        t.ranks.deinit(gpa);
    }

    /// Encodes `text` to token ids. Caller owns the result.
    pub fn encode(t: *const Tokenizer, gpa: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var bounds: std.ArrayList(usize) = .empty;
        defer bounds.deinit(gpa);

        var i: usize = 0;
        while (i < text.len) {
            const end = nextChunk(text, i);
            try t.encodeChunk(gpa, text[i..end], &bounds, &out);
            i = end;
        }
        return out.toOwnedSlice(gpa);
    }

    fn encodeChunk(t: *const Tokenizer, gpa: Allocator, chunk: []const u8, bounds: *std.ArrayList(usize), out: *std.ArrayList(u32)) !void {
        if (t.ranks.get(chunk)) |id| return out.append(gpa, id);
        // bounds[k]..bounds[k+1] is part k; start with single bytes.
        bounds.clearRetainingCapacity();
        for (0..chunk.len + 1) |k| try bounds.append(gpa, k);
        while (bounds.items.len > 2) {
            var best: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            for (0..bounds.items.len - 2) |k| {
                const pair = chunk[bounds.items[k]..bounds.items[k + 2]];
                if (t.ranks.get(pair)) |r| if (r < best_rank) {
                    best_rank = r;
                    best = k;
                };
            }
            const k = best orelse break;
            _ = bounds.orderedRemove(k + 1);
        }
        for (0..bounds.items.len - 1) |k| {
            const part = chunk[bounds.items[k]..bounds.items[k + 1]];
            if (t.ranks.get(part)) |id| try out.append(gpa, id);
        }
    }
};

fn isLetter(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c >= 0x80;
}
fn isSpace(c: u8) bool {
    return std.ascii.isWhitespace(c);
}

/// GPT-2 pre-tokenizer (ASCII approximation of its regex):
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn nextChunk(s: []const u8, i: usize) usize {
    if (s[i] == '\'') {
        for ([_][]const u8{ "re", "ve", "ll", "s", "t", "m", "d" }) |suf| {
            if (std.mem.startsWith(u8, s[i + 1 ..], suf)) return i + 1 + suf.len;
        }
    }
    var j = i;
    if (s[j] == ' ' and j + 1 < s.len and !isSpace(s[j + 1])) j += 1;
    const c = s[j];
    if (isLetter(c)) {
        while (j < s.len and isLetter(s[j])) j += 1;
        return j;
    }
    if (std.ascii.isDigit(c)) {
        while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
        return j;
    }
    if (!isSpace(c)) {
        while (j < s.len and !isSpace(s[j]) and !isLetter(s[j]) and !std.ascii.isDigit(s[j])) j += 1;
        return j;
    }
    // Whitespace run; leave the last space to prefix a following word.
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j < s.len and j - i > 1) return j - 1;
    return j;
}
