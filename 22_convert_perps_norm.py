"""Fast, filtered converter: provider's *.bin.zst shards -> one Parquet per symbol per UTC day (best bid/offer).

Two layouts are handled (detected from the first bytes):
  BINARY (version 2 and 3 files: Binance2026MMDDperpN, NN_perp_filtered, perps_norm_*): 32-byte header
      u16 version, u16 type, u32 payload length, i64 receive time (ns), 16 more bytes; payload starts with
      u32 product_pid (14Jul_perp_id.csv), u32, i64 seq, i64 local time (ns), i64 receive time (ns),
      i64 exchange time (ns, ms precision), i64 update id, then
        type 1 (80 bytes)  best bid/offer: bid px, bid qty, ask px, ask qty   (int64, scaled by 1e9)
        type 2 (72 bytes)  trade                                               (skipped)
        type 3 (576 bytes) 16-level book snapshot                              (best level kept as a fallback)
  JSON (version 1): 32-byte header + Binance bookTicker JSON.
Only the paper's 10 perps are kept; the raw bytes are searched for their ids, so this is much faster than
parsing every packet.  Exchange time is kept (T_ms); the lead-lag results need exchange time.
  python 22_convert_perps_norm.py "/content/drive/MyDrive/perps/*.bin.zst" out_dir [stop_after_GB]
Output: out_dir/<SYMBOL>/<YYYY-MM-DD>/<source>-part-*.parquet
  (columns: recv_ns, T_ms, E_ms, e, u, bid, bid_qty, ask, ask_qty;  e = "bbo" (type 1), "book16" (type 3) or
   "bookTicker" (JSON))
"""
import glob, json, os, re, struct, sys, time
import numpy as np
import pandas as pd
import zstandard as zstd

SYMBOLS = ["XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "COPPERUSDT", "CLUSDT", "BZUSDT",
           "NATGASUSDT", "SPYUSDT", "BTCUSDT"]
PIDS = {4449: "BTCUSDT", 4986: "BZUSDT", 4987: "CLUSDT", 4989: "COPPERUSDT", 4999: "NATGASUSDT",
        5005: "SPYUSDT", 5008: "XAGUSDT", 5009: "XAUUSDT", 5010: "XPDUSDT", 5011: "XPTUSDT"}
SCALE = 1e-9
PAT = re.compile(rb'"s": ?"(' + b"|".join(x.encode() for s in SYMBOLS for x in (s, s.lower())) + rb')"')
SIG1 = b"\x00\x01\x00\x50\x00\x00\x00"          # (after the version byte) type 1, payload 80 bytes
SIG3 = b"\x00\x03\x00\x40\x02\x00\x00"          # type 3, payload 576 bytes
CHUNK = 64 * 1024 * 1024
TAIL = 16 * 1024          # bytes kept for the next chunk (a packet is < 1 kB)
OV = 1024                 # overlap kept before the cut so a packet's header and start stay visible
FLUSH = 500_000
MAX_GB = None             # set by the 3rd command-line argument: stop after this many GB (quick peek)
GIVE_UP_GB = 3            # skip a file if none of the 10 perps turns up in its first GIVE_UP_GB GB


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Sink:
    def __init__(self, out):
        self.out, self.rows, self.parts, self.tag = out, [], 0, "x"

    def add(self, r):
        self.rows.append(r)
        if len(self.rows) >= FLUSH:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        d = pd.DataFrame(self.rows, columns=["symbol", "recv_ns", "T_ms", "E_ms", "e", "u", "bid", "bid_qty", "ask", "ask_qty"])
        dn = (d["T_ms"].astype("float64").fillna(d["E_ms"].astype("float64")).fillna(d["recv_ns"] // 1_000_000) // 86_400_000).astype("int64")
        names = {x: time.strftime("%Y-%m-%d", time.gmtime(x * 86400)) for x in dn.unique()}
        d["date"] = dn.map(names)
        for (sym, day), g in d.groupby(["symbol", "date"]):
            p = os.path.join(self.out, sym, day)
            os.makedirs(p, exist_ok=True)
            g.drop(columns=["symbol", "date"]).to_parquet(os.path.join(p, f"{self.tag}-part-{self.parts:05d}.parquet"), index=False)
        self.parts += 1
        self.rows = []


class _nullctx:
    def __init__(self, f): self.f = f
    def __enter__(self): return self.f
    def __exit__(self, *a): return False


def hexdump(b):
    for off in range(0, len(b), 32):
        c = b[off:off + 32]
        print(f"    {off:04x}  {c.hex(' ')}  {''.join(chr(x) if 32 <= x < 127 else '.' for x in c)}", flush=True)


def utc(ms):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000))


# ------------------------------------------------------------------ binary layout (versions 2 and 3)
PID_ARR = np.array(sorted(PIDS), dtype=np.uint32)


def find_headers(arr, sig, rare):
    """positions h of packet headers whose bytes h+1.. equal sig (vectorised; starts from a rare byte of sig)"""
    idx = np.flatnonzero(arr == sig[rare]) - rare
    idx = idx[(idx >= 0) & (idx + 36 <= len(arr))]
    for k, b in enumerate(sig):
        if k != rare:
            idx = idx[arr[idx + k] == b]
    return idx - 1                                   # the version byte comes first


def scan_binary(data, cut, done_upto, sink, st):
    arr = np.frombuffer(data, dtype=np.uint8)
    for sig, rare in ((SIG1, 3), (SIG3, 4)):
        h = find_headers(arr, sig, rare)
        h = h[(h >= 0) & (h + 36 <= cut) & (h + 36 > done_upto)]
        if not len(h):
            continue
        h = h[(arr[h] == 2) | (arr[h] == 3)]
        pid = (arr[h + 32].astype(np.uint32) | (arr[h + 33].astype(np.uint32) << 8)
               | (arr[h + 34].astype(np.uint32) << 16) | (arr[h + 35].astype(np.uint32) << 24))
        h = h[np.isin(pid, PID_ARR)]
        for x in h.tolist():
            typ, plen = data[x + 2], struct.unpack_from("<I", data, x + 4)[0]
            if x + 32 + plen > len(data):
                st["bad"] += 1
                continue
            pid_, _, _, t_loc, t_rcv, t_exc, u = struct.unpack_from("<IIqqqqq", data, x + 32)
            sym = PIDS[pid_]
            if typ == 1:
                bp, bq, ap, aq = struct.unpack_from("<qqqq", data, x + 32 + 48)
                e = "bbo"
            else:
                bp, bq = struct.unpack_from("<qq", data, x + 32 + 64)
                ap, aq = struct.unpack_from("<qq", data, x + 32 + 64 + 16 * 16)
                e = "book16"
            if not (0 < bp < ap):
                st["bad"] += 1
                continue
            t_ms = t_exc // 1_000_000 if t_exc > 0 else None
            key = (sym, e)
            st["counts"][key] = st["counts"].get(key, 0) + 1
            if st["counts"][key] == 1:
                print(f"  first {sym} {e}: {utc(t_ms or t_loc // 1_000_000)} UTC  bid {bp * SCALE:,.6g} x {bq * SCALE:,.6g}"
                      f"  ask {ap * SCALE:,.6g} x {aq * SCALE:,.6g}", flush=True)
            sink.add((sym, t_rcv, t_ms, t_loc // 1_000_000, e, u, bp * SCALE, bq * SCALE, ap * SCALE, aq * SCALE))
            st["kept"] += 1


def types_seen(data):
    return {f"v{v} type{t}": data.count(bytes([v, 0, t, 0])) for v in (2, 3) for t in (1, 2, 3)}


# ------------------------------------------------------------------ JSON layout (version 1)
def scan_json(data, cut, done_upto, sink, st):
    for m in PAT.finditer(data, 0, cut):
        if m.end() <= done_upto:
            continue
        s0, h, lo_ = m.start(), -1, max(0, m.start() - 1024)
        for _ in range(3):                       # the payload may wrap the message ({"stream":..,"data":{..}})
            s0 = data.rfind(b"{", lo_, s0)
            if s0 < 32:
                break
            if data[s0 - 32] == 0x01 and data[s0 - 31] == 0x00:
                h = s0 - 32
                break
        if h < 0:
            st["bad"] += 1
            continue
        plen = struct.unpack_from("<I", data, h + 4)[0]
        if plen == 0 or plen > 10000 or s0 + plen > len(data):
            st["bad"] += 1
            continue
        raw = data[s0:s0 + plen]
        if raw.endswith(b"}}") and raw.count(b"{") < raw.count(b"}"):
            raw = raw[:-1]
        try:
            o = json.loads(raw)
        except ValueError:
            st["bad"] += 1
            continue
        if isinstance(o.get("data"), dict):
            o = o["data"]
        if not (isinstance(o.get("b"), str) and isinstance(o.get("a"), str)):
            continue                              # trade, depth, ...
        sym = str(o.get("s")).upper(); key = (sym, "bookTicker")
        st["counts"][key] = st["counts"].get(key, 0) + 1
        if st["counts"][key] == 1:
            print(f"  first {sym}: {raw[:200].decode('utf-8', 'replace')}", flush=True)
        sink.add((sym, struct.unpack_from("<q", data, h + 8)[0], o.get("T"), o.get("E"),
                  o.get("e"), o.get("u"), num(o.get("b")), num(o.get("B")), num(o.get("a")), num(o.get("A"))))
        st["kept"] += 1


def convert(path, sink, fobj=None):
    """path = a local file, or just a name when fobj (any object with .read(n), e.g. a Drive download stream) is given"""
    sink.flush()
    sink.tag, sink.parts = os.path.basename(path).split(".")[0], 0     # output names carry the source file
    flag = os.path.join(sink.out, sink.tag + "-done.flag")
    if os.path.exists(flag) and open(flag).read().strip() not in ("", "0"):
        print("already converted, skipping", path, flush=True)
        return
    t0, nbytes, done_upto, peek, scan, types, head = time.time(), 0, 0, False, None, {}, b""
    st = dict(kept=0, bad=0, counts={})
    with (open(path, "rb") if fobj is None else _nullctx(fobj)) as f:
        reader = zstd.ZstdDecompressor().stream_reader(f, read_size=CHUNK, read_across_frames=True)
        left = b""
        while True:
            chunk = reader.read(CHUNK)
            last = not chunk
            data = left + chunk
            nbytes += len(chunk)
            if scan is None:
                head = data[:128]
                if len(data) >= 2 and data[0] in (2, 3) and data[1] == 0:
                    scan = scan_binary
                    print(f"  binary layout, version {data[0]}", flush=True)
                else:
                    scan = scan_json
                    print("  JSON layout", flush=True)
            if scan is scan_binary and not types:
                types = types_seen(chunk)
            cut = len(data) if last else max(len(data) - TAIL, 0)
            scan(data, cut, done_upto, sink, st)
            if last:
                break
            if MAX_GB and nbytes >= MAX_GB * 2**30:
                peek = True
                break
            start = max(cut - OV, 0)
            left, done_upto = data[start:], cut - start
            if nbytes // (1 << 30) != (nbytes - len(chunk)) // (1 << 30):
                el = time.time() - t0
                per = ", ".join(f"{k[0].replace('USDT', '')} {v:,}" for k, v in sorted(st["counts"].items()))
                print(f"  {nbytes / 2**30:6.1f} GB, {st['kept']:,} rows kept [{per}], {nbytes / 2**20 / el:,.0f} MB/s", flush=True)
                if nbytes < 2**30 + CHUNK and types:
                    print("  message headers in the first 64 MB (approx.):", {k: f"{v:,}" for k, v in types.items() if v > 100}, flush=True)
                if st["kept"] == 0 and nbytes >= GIVE_UP_GB * 2**30:
                    print(f"  GIVING UP on this file: none of the 10 perps in the first {GIVE_UP_GB} GB. First bytes:", flush=True)
                    hexdump(head)
                    return
    sink.flush()
    os.makedirs(sink.out, exist_ok=True)
    if peek:                                   # quick look only: do not mark the file as converted
        print(f"  PEEK ONLY (first {MAX_GB:g} GB) - delete this output folder before the full run", flush=True)
    elif st["kept"]:
        open(flag, "w").write(str(st["kept"]))
    print(f"done {os.path.basename(path)}: {st['kept']:,} rows kept, {st['bad']:,} unreadable, {time.time() - t0:,.0f} s", flush=True)
    if st["counts"]:
        print("  rows per symbol:", ", ".join(f"{k[0]} {k[1]} {v:,}" for k, v in sorted(st["counts"].items())), flush=True)


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else "data/native_provider"
    MAX_GB = float(sys.argv[3]) if len(sys.argv) > 3 else None
    if not files:
        sys.exit("no files match " + sys.argv[1])
    sink = Sink(out)
    for f in files:
        print("converting", f, flush=True)
        convert(f, sink)
