// Shared helpers for the Syntaxless README films, v3: easing vocabulary, a seeded RNG, a driver
// whose render is a pure function of time, a text field with a human typist, and the pointer.
(function () {
  gsap.registerPlugin(CustomEase);
  // GSAP's default moves elements with translate3d while they animate, which promotes them to
  // their own compositor layer; text on such a layer loses subpixel antialiasing and keeps it
  // lost after the move, so a loop's last frame stops matching its first. Stay 2D.
  gsap.config({ force3D: false });

  // One small motion vocabulary, so the films feel made by one hand. Curves stay gentle: a steep
  // curve crams the travel into a few frames, and at 60fps that reads as stutter. None of these
  // peaks above ~2.9x its average speed.
  CustomEase.create("sx.out", "0.25, 0.5, 0.25, 1"); // arrivals: brisk start, long settle
  CustomEase.create("sx.in", "0.55, 0, 0.9, 0.45"); // departures: gather, then leave
  CustomEase.create("sx.move", "0.45, 0, 0.25, 1"); // travel: soft start, soft landing
  CustomEase.create("sx.swap", "0.65, 0, 0.35, 1"); // in-place state changes

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  const smooth = (x) => {
    const c = clamp01(x);
    return c * c * (3 - 2 * c);
  };

  /** Adds one driver tween whose onUpdate receives the timeline time; also renders t = 0. */
  function drive(tl, duration, render) {
    const clock = { t: 0 };
    tl.to(clock, { t: duration, duration, ease: "none", onUpdate: () => render(clock.t) }, 0);
    render(0);
  }

  /**
   * A number that changes over time: a list of [start, from, to, duration] steps, eased.
   * value(t) is a pure function of t.
   */
  function counter(steps, initial) {
    return function value(t) {
      let v = initial;
      for (const [at, from, to, d] of steps) {
        if (t < at) break;
        v = from + (to - from) * smooth(d > 0 ? (t - at) / d : 1);
      }
      return v;
    };
  }

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /**
   * A text field whose contents at time t are computed from a schedule of keystrokes. Build the
   * schedule with focus/type/selectAll/clear/blur, then call render(t) from the driver.
   */
  class Field {
    constructor(el, opts) {
      this.text = el.querySelector(".text");
      this.ph = el.querySelector(".ph");
      this.keys = [[-Infinity, ""]];
      this.focusSpans = [];
      this.selections = [];
      this.random = rng((opts && opts.seed) || 7);
      this.caretClass = opts && opts.bar ? "caret bar" : "caret";
      this.last = "";
    }

    focus(t) {
      this.focusSpans.push([t, Infinity]);
      return t;
    }

    blur(t) {
      const open = this.focusSpans[this.focusSpans.length - 1];
      if (open) open[1] = t;
      return t;
    }

    /** Types `s` from time t at a human cadence: bursts in words, small pauses at spaces. */
    type(t, s, speed) {
      const base = 0.043 / (speed || 1);
      let now = t;
      let text = this.keys[this.keys.length - 1][1];
      for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        let gap = base * (0.55 + this.random() * 0.8);
        if (ch === " ") gap *= 1.5;
        if (s[i - 1] === ",") gap *= 1.6;
        if (this.random() < 0.05) gap += 0.08 + this.random() * 0.07;
        now += gap;
        text += ch;
        this.keys.push([now, text]);
      }
      return now;
    }

    /** Select all (shown as a selection for `hold` seconds), then delete. */
    replace(t, hold) {
      const h = hold === undefined ? 0.22 : hold;
      this.selections.push([t, t + h]);
      this.keys.push([t + h, ""]);
      return t + h;
    }

    /** Empties the field at t without showing it (for resets while the field is hidden). */
    set(t, s) {
      this.keys.push([t, s]);
      return t;
    }

    render(t) {
      let text = "";
      let lastKey = -Infinity;
      for (const [kt, kv] of this.keys) {
        if (kt <= t) {
          text = kv;
          lastKey = kt;
        }
      }
      const focused = this.focusSpans.find(([a, b]) => t >= a && t < b);
      const selected = this.selections.some(([a, b]) => t >= a && t < b);
      let caret = "";
      if (focused) {
        const idle = t - Math.max(lastKey, focused[0]);
        const on = idle < 0.5 || (idle - 0.5) % 1.06 < 0.53;
        caret = '<span class="' + this.caretClass + '" style="opacity:' + (on && !selected ? 1 : 0) + '"></span>';
      }
      const body = selected ? '<span class="sel">' + esc(text) + "</span>" : esc(text);
      const key = body + caret;
      if (key === this.last) return;
      this.last = key;
      this.text.innerHTML = body + caret;
      this.ph.style.display = text ? "none" : "";
    }
  }

  /** Window controls for a title bar: a grey set, and a coloured set shown while focused. */
  const LIGHTS =
    '<span class="lights"><span class="idle"><i></i><i></i><i></i></span>' +
    '<span class="live"><i></i><i></i><i></i></span></span>';

  /** Adds the mouse pointer to `parent` and returns it. */
  function pointer(parent) {
    const el = document.createElement("div");
    el.className = "pointer";
    el.innerHTML =
      '<svg viewBox="0 0 34 50" width="34" height="50">' +
      '<path d="M4 3 L4 38 L12.5 30 L18.5 44.5 L25 41.8 L19 27.8 L30.5 27.8 Z" ' +
      'fill="#111318" stroke="#ffffff" stroke-width="2.6" stroke-linejoin="round"/></svg>';
    parent.appendChild(el);
    return el;
  }

  /**
   * Choreography for a pointer. Its tip sits at (x, y). Moves travel on a slight arc, because x
   * and y ease differently, the way a hand moves a mouse. Long moves get more time.
   */
  const TIP = [4, 3];
  function hand(tl, el) {
    let at = [0, 0];
    return {
      place(t, x, y) {
        at = [x, y];
        tl.set(el, { x: x - TIP[0], y: y - TIP[1] }, t);
      },
      move(t, x, y, duration) {
        const dist = Math.hypot(x - at[0], y - at[1]);
        const d = duration || Math.min(0.95, 0.4 + dist / 1500);
        at = [x, y];
        tl.to(el, { x: x - TIP[0], duration: d, ease: "sx.move" }, t);
        tl.to(el, { y: y - TIP[1], duration: d, ease: "sx.swap" }, t);
        return t + d;
      },
      click(t) {
        tl.to(el, { scale: 0.85, duration: 0.07, ease: "sx.swap" }, t);
        tl.to(el, { scale: 1, duration: 0.16, ease: "sx.out" }, t + 0.07);
        return t + 0.23;
      },
    };
  }


  // ---------- Code ----------

  // Highlighting kept nearly monochrome: the English inside strings is the brightest thing on
  // screen, code around it recedes. `{{id}}` becomes an empty, typeable slot (see slot()).
  const KEYWORDS = new Set(
    "import from const let await async if else for of return export new throw type interface".split(" "),
  );
  function slot(id) {
    return '<span class="slot" id="' + id + '"><span class="ph"></span><span class="text"></span></span>';
  }
  function hl(src) {
    const out = [];
    const re =
      /(\{\{[\w-]+\}\})|(\/\/.*$)|("(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)|(\b\d[\d_.]*\b)|([A-Za-z_$][\w$]*)|(=>|[=!]==?|\?\?|\?\.|&&|\|\||[+\-*/<>]=?|=)|([()[\]{},.;:?!])|(\s+)|(.)/g;
    let m;
    while ((m = re.exec(src))) {
      const [tok, ph, com, str, num, id, op, pun, ws] = m;
      if (ph) out.push(slot(ph.slice(2, -2)));
      else if (com) out.push('<span class="cm">' + esc(com) + "</span>");
      else if (str) out.push('<span class="st">' + esc(str).replace(/\{\{([\w-]+)\}\}/g, (_, k) => slot(k)) + "</span>");
      else if (num) out.push('<span class="nu">' + num + "</span>");
      else if (id) out.push(KEYWORDS.has(id) ? '<span class="kw">' + id + "</span>" : id);
      else if (op || pun) out.push('<span class="pu">' + esc(op || pun) + "</span>");
      else if (ws) out.push(ws);
      else out.push(esc(tok));
    }
    return out.join("");
  }

  /** Fills `el` with numbered code lines; returns the line elements. */
  function code(el, lines) {
    el.innerHTML = lines
      .map((src, i) => '<div class="ln"><span class="hl"></span><span class="no">' + (i + 1) + '</span><span class="src">' + hl(src) + "</span></div>")
      .join("");
    return Array.from(el.children);
  }

  // ---------- Terminal ----------

  /**
   * A terminal whose contents at time t are computed from a schedule of keystrokes and outputs.
   * Build it with term.at(t).type(...).enter(lines), then call term.render(t) from the driver.
   * Output lines are HTML (escape text with SX.esc).
   */
  class Terminal {
    constructor(body, opts) {
      this.body = body;
      this.prompt = opts.prompt || '<span class="prompt">❯ </span>';
      this.rows = opts.rows || 10;
      this.lineHeight = opts.lineHeight || 40;
      this.seed = opts.seed || 7;
      this.entries = [];
      this.clears = [];
      this.last = "";
    }

    /** Starts a command line at time t; returns a chainable typist. */
    at(t) {
      const term = this;
      const entry = { kind: "cmd", at: t, keys: [[t, ""]], enter: Infinity, done: Infinity };
      this.entries.push(entry);
      const random = rng(this.seed++ * 7919);
      let now = t;
      let text = "";
      const typist = {
        type(s, speed) {
          const base = 0.043 / (speed || 1);
          for (let i = 0; i < s.length; i++) {
            let gap = base * (0.55 + random() * 0.8);
            if (s[i] === " ") gap *= 1.5;
            if (random() < 0.05) gap += 0.09 + random() * 0.08;
            now += gap;
            text += s[i];
            entry.keys.push([now, text]);
          }
          return typist;
        },
        /** Recalls a previous command from history (one keypress). */
        recall(s, delay) {
          now += delay === undefined ? 0.14 : delay;
          text = s;
          entry.keys.push([now, text]);
          return typist;
        },
        pause(s) {
          now += s;
          return typist;
        },
        /** Presses enter after a short beat; `lines` print `delay` seconds later. */
        enter(lines, delay) {
          now += 0.26;
          entry.enter = now;
          const printAt = now + (delay === undefined ? 0.12 : delay);
          term.entries.push({ kind: "out", at: printAt, lines: lines || [] });
          entry.done = printAt;
          now = printAt;
          return typist;
        },
        get end() {
          return now;
        },
      };
      return typist;
    }

    /** Clears the screen at time t, as Ctrl-L would. */
    clear(t) {
      this.clears.push(t);
    }

    lineAt(t) {
      const lines = [];
      let lastClear = -Infinity;
      for (const c of this.clears) if (c <= t) lastClear = Math.max(lastClear, c);
      let active = false;
      let idleSince = 0;
      const ordered = this.entries.slice().sort((a, b) => a.at - b.at);
      for (const entry of ordered) {
        if (entry.at > t) break;
        if (entry.kind === "cmd") {
          let text = "";
          let lastKey = entry.at;
          for (const [kt, kv] of entry.keys) {
            if (kt <= t) {
              text = kv;
              lastKey = kt;
            }
          }
          const typing = t < entry.enter;
          if (typing) active = true;
          if (entry.at >= lastClear || typing) lines.push({ html: this.prompt + esc(text), caret: typing, lastKey });
          if (!typing) idleSince = entry.done;
        } else if (entry.at >= lastClear) {
          for (const line of entry.lines) lines.push({ html: line });
        }
      }
      if (!active) lines.push({ html: this.prompt, caret: true, lastKey: Math.max(idleSince, lastClear) });
      return lines;
    }

    render(t) {
      const lines = this.lineAt(t);
      const visible = lines.slice(Math.max(0, lines.length - this.rows));
      const html = visible
        .map((line) => {
          let caret = "";
          if (line.caret) {
            const idle = t - line.lastKey;
            const on = idle < 0.5 || (idle - 0.5) % 1.06 < 0.53;
            caret = '<span class="caret" style="opacity:' + (on ? 1 : 0) + '"></span>';
          }
          return '<div class="term-line">' + line.html + caret + "</div>";
        })
        .join("");
      if (html === this.last) return;
      this.last = html;
      this.body.innerHTML = html;
    }
  }

  /**
   * The package card every package film carries: its name, its gzip size against the 100 kB
   * budget, and where it runs. Values are measured: the whole built package at gzip -9.
   */
  function packageCard(el, name, kb) {
    el.innerHTML =
      '<div class="panel-head">@syntaxless/' + name + '<span class="tag">package</span></div>' +
      '<div class="pkg-body">' +
      '<div class="pkg-label">gzip, model included</div>' +
      '<div class="pkg-value">' + kb.toFixed(1) + ' kB</div>' +
      '<div class="pkg-bar"><i style="width:' + kb + '%"></i></div>' +
      '<div class="pkg-scale"><span>0</span><span>100 kB</span></div>' +
      '<div class="pkg-facts"><div>0 runtime dependencies</div><div>Node.js · Bun · browser</div><div>runs offline, no API key</div></div>' +
      "</div><div class=\"focus\"></div>";
    return el.querySelector(".pkg-bar i");
  }

  window.SX = { rng, clamp01, smooth, drive, counter, esc, Field, LIGHTS, pointer, hand, slot, hl, code, Terminal, packageCard };
})();
