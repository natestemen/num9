// Running engine work off the page: a pool of workers for hints (many small
// independent jobs), and single workers for long jobs that are simply
// thrown away when no longer wanted.

const WORKER = new URL("./worker.js", import.meta.url);
let nextId = 1;

function spawn() {
  return new Worker(WORKER, { type: "module" });
}

/// One worker running one long job at a time (with progress messages).
export class Background {
  constructor() {
    this.worker = null;
  }

  /// Run `msg`; `onMessage(reply)` hears every reply, including the final
  /// one (type "done"). Returns a promise of the final reply.
  run(msg, onMessage = () => {}) {
    this.worker ??= spawn();
    const id = nextId++;
    this.id = id;
    return new Promise((resolve, reject) => {
      const listen = ({ data }) => {
        if (data.id !== id) return;
        if (data.type === "error") reject(new Error(data.message));
        onMessage(data);
        if (data.type === "done" || data.type === "error") {
          this.worker?.removeEventListener("message", listen);
          resolve(data);
        }
      };
      this.worker.addEventListener("message", listen);
      this.worker.postMessage({ ...msg, id });
    });
  }

  /// Abandon whatever is running (the worker is replaced on next use).
  stop() {
    this.worker?.terminate();
    this.worker = null;
  }
}

/// Workers sharing a queue of short jobs.
export class Pool {
  constructor(size) {
    this.workers = Array.from({ length: size }, spawn);
    this.idle = [...this.workers];
    this.queue = [];
    this.handlers = new Map();
    for (const w of this.workers) {
      w.onmessage = ({ data }) => {
        const job = this.handlers.get(data.id);
        this.handlers.delete(data.id);
        this.idle.push(w);
        if (job && !job.cancelled.value) job.resolve(data);
        this.pump();
      };
    }
  }

  static default() {
    const cores = navigator.hardwareConcurrency || 4;
    return new Pool(Math.max(1, Math.min(8, cores - 1)));
  }

  /// Queue jobs; `onResult(reply)` hears each as it finishes. Returns a
  /// function that cancels whatever hasn't run yet.
  submit(msgs, onResult) {
    const cancelled = { value: false };
    for (const msg of msgs) {
      const id = nextId++;
      this.queue.push({ msg: { ...msg, id }, cancelled, resolve: onResult });
    }
    this.pump();
    return () => {
      cancelled.value = true;
      this.queue = this.queue.filter((j) => j.cancelled !== cancelled);
    };
  }

  pump() {
    while (this.idle.length && this.queue.length) {
      const job = this.queue.shift();
      const w = this.idle.pop();
      this.handlers.set(job.msg.id, job);
      w.postMessage(job.msg);
    }
  }
}
