<script>
  import { onMount, tick } from 'svelte'

  let question = ''
  let mode = 'hybrid'
  let useReranker = true
  let loading = false
  let result = null
  let error = ''
  let health = null
  let active = null          // แหล่งอ้างอิงที่กำลังถูกเน้น
  let expanded = {}

  const examples = [
    'องค์ประชุมของรัฐสภาต้องมีสมาชิกเท่าใด',
    'การประชุมลับทำได้ในกรณีใดบ้าง',
    'ญัตติที่ไม่ต้องเสนอล่วงหน้าเป็นหนังสือมีอะไรบ้าง'
  ]

  onMount(async () => {
    await poll()
  })

  async function poll() {
    try {
      health = await (await fetch('/api/health')).json()
      if (!health.ready && !health.error) setTimeout(poll, 2000)
    } catch (e) {
      health = { ready: false, error: 'เชื่อมต่อ backend ไม่ได้ — ตรวจว่ารัน uvicorn อยู่หรือไม่' }
      setTimeout(poll, 3000)
    }
  }

  async function ask() {
    if (!question.trim() || loading) return
    loading = true; error = ''; result = null; active = null; expanded = {}
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode, use_reranker: useReranker })
      })
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`)
      result = await res.json()
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  function onKey(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask()
  }

  // แยกข้อความคำตอบออกเป็นชิ้น เพื่อทำให้ [S1] กดได้
  function segments(text) {
    const out = []
    const re = /\[S(\d+)\]/g
    let last = 0, m
    while ((m = re.exec(text))) {
      if (m.index > last) out.push({ t: text.slice(last, m.index) })
      out.push({ cite: Number(m[1]) })
      last = m.index + m[0].length
    }
    if (last < text.length) out.push({ t: text.slice(last) })
    return out
  }

  async function focusSource(n) {
    active = n
    await tick()
    document.getElementById(`src-${n}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }

  function shift(s) {
    if (s.prev_rank == null || s.prev_rank === s.rank) return null
    return s.prev_rank - s.rank      // บวก = ถูกดันขึ้น
  }
</script>

<header>
  <h1>ค้นกฎหมายไทย</h1>
  {#if health?.ready}
    <p class="meta">
      {health.corpus.documents} ฉบับ · {health.corpus.chunks.toLocaleString()} ท่อนข้อความ ·
      ตัดที่ {health.config.chunk_size} token
    </p>
  {:else if health?.error}
    <p class="meta warn">{health.error}</p>
  {:else}
    <p class="meta">กำลังโหลดโมเดลเข้าหน่วยความจำ</p>
  {/if}
</header>

<main>
  <section class="ask">
    <textarea
      bind:value={question}
      on:keydown={onKey}
      rows="2"
      placeholder="พิมพ์คำถามเกี่ยวกับกฎหมายในคลัง"
      disabled={!health?.ready}></textarea>

    <div class="controls">
      <div class="knobs">
        <label>
          วิธีค้น
          <select bind:value={mode} disabled={!health?.ready}>
            <option value="hybrid">ผสม dense + BM25</option>
            <option value="dense">dense อย่างเดียว</option>
            <option value="bm25">BM25 อย่างเดียว</option>
          </select>
        </label>
        <label class="check">
          <input type="checkbox" bind:checked={useReranker} disabled={!health?.ready} />
          จัดอันดับซ้ำด้วย cross-encoder
        </label>
      </div>
      <button class="go" on:click={ask} disabled={loading || !health?.ready || !question.trim()}>
        {loading ? 'กำลังค้น' : 'ค้นคำตอบ'}
      </button>
    </div>

    {#if !result && !loading}
      <ul class="examples">
        {#each examples as ex}
          <li><button on:click={() => { question = ex; ask() }}>{ex}</button></li>
        {/each}
      </ul>
    {/if}
  </section>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if loading}
    <p class="status">กำลังค้นในคลังเอกสาร…</p>
  {/if}

  {#if result}
    <div class="grid">
      <article class="answer">
        <p class="stamp">
          {result.mode === 'hybrid' ? 'ผสม dense + BM25' : result.mode}
          {#if result.reranked}· จัดอันดับซ้ำแล้ว{/if}
          · ค้น {result.timings.retrieval_s}s · ตอบ {result.timings.generation_s}s
        </p>
        {#each result.answer.split('\n').filter(Boolean) as para}
          <p>
            {#each segments(para) as seg}
              {#if seg.cite}<button
                  class="cite"
                  class:on={active === seg.cite}
                  on:click={() => focusSource(seg.cite)}>S{seg.cite}</button>
              {:else}{seg.t}{/if}
            {/each}
          </p>
        {/each}
      </article>

      <aside class="sources">
        <h2>ที่มาของคำตอบ</h2>
        {#each result.sources as s, i}
          {@const d = shift(s)}
          <div class="src" id={`src-${i + 1}`} class:on={active === i + 1}>
            <div class="srchead">
              <span class="tag">S{i + 1}</span>
              <span class="cit">{s.section_no || s.title}</span>
              {#if d !== null}
                <span class="shift" class:up={d > 0} class:down={d < 0}>
                  {s.prev_rank} ▸ {s.rank}
                </span>
              {/if}
            </div>
            <p class="crumb">{s.title}{s.breadcrumb ? ' · ' + s.breadcrumb : ''}</p>
            <div class="channels">
              {#if s.found_by.includes('dense')}<span class="ch dense">พบด้วย dense</span>{/if}
              {#if s.found_by.includes('bm25')}<span class="ch lex">พบด้วย BM25</span>{/if}
              <span class="score">{s.score.toFixed(4)}</span>
            </div>
            <p class="body" class:clip={!expanded[i]}>{s.text}</p>
            <button class="more" on:click={() => expanded[i] = !expanded[i]}>
              {expanded[i] ? 'ย่อ' : 'อ่านเต็ม'}
            </button>
          </div>
        {/each}
      </aside>
    </div>
  {/if}
</main>

<style>
  header {
    border-bottom: 1px solid var(--rule);
    padding: 2rem 1.5rem 1.25rem;
    max-width: 1180px;
    margin: 0 auto;
  }
  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0;
  }
  .meta {
    margin: .25rem 0 0;
    color: var(--ink-soft);
    font-size: .9rem;
    font-variant-numeric: tabular-nums;
  }
  .warn { color: var(--seal); }

  main { max-width: 1180px; margin: 0 auto; padding: 1.75rem 1.5rem 5rem; }

  .ask textarea {
    width: 100%;
    padding: .85rem 1rem;
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-radius: 3px;
    resize: vertical;
    line-height: 1.6;
  }
  .ask textarea:focus { border-color: var(--ink-soft); outline: none; }

  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: .75rem;
  }
  .knobs { display: flex; gap: 1.25rem; align-items: center; flex-wrap: wrap; font-size: .9rem; color: var(--ink-soft); }
  .knobs select {
    margin-left: .4rem;
    padding: .3rem .5rem;
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-radius: 3px;
    color: var(--ink);
  }
  .check { display: flex; align-items: center; gap: .45rem; cursor: pointer; }
  .check input { accent-color: var(--seal); }

  .go {
    padding: .55rem 1.6rem;
    background: var(--seal);
    color: var(--paper-raised);
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-weight: 600;
  }
  .go:disabled { background: var(--rule); color: var(--ink-soft); cursor: not-allowed; }

  .examples { list-style: none; padding: 0; margin: 1.5rem 0 0; }
  .examples li { border-top: 1px solid var(--rule); }
  .examples button {
    width: 100%;
    text-align: left;
    padding: .7rem .2rem;
    background: none;
    border: none;
    cursor: pointer;
    color: var(--ink-soft);
  }
  .examples button:hover { color: var(--seal); }

  .status, .error { margin-top: 2rem; color: var(--ink-soft); }
  .error { color: var(--seal); }

  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
    gap: 2.5rem;
    margin-top: 2.5rem;
    align-items: start;
  }

  .answer { max-width: var(--measure); }
  .answer p { margin: 0 0 1rem; }
  .stamp {
    color: var(--ink-soft);
    font-size: .82rem;
    font-variant-numeric: tabular-nums;
    padding-bottom: .6rem;
    border-bottom: 1px solid var(--rule);
  }

  .cite {
    font-family: var(--font-data);
    font-size: .74rem;
    padding: .05rem .3rem;
    margin: 0 .12rem;
    background: none;
    border: 1px solid var(--rule);
    border-radius: 2px;
    color: var(--seal);
    cursor: pointer;
    vertical-align: .08em;
  }
  .cite:hover, .cite.on { background: var(--seal); color: var(--paper-raised); border-color: var(--seal); }

  .sources h2 {
    font-size: .95rem;
    font-weight: 600;
    margin: 0 0 .9rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid var(--rule);
  }
  .src {
    padding: .9rem 0;
    border-bottom: 1px solid var(--rule);
  }
  .src.on { background: var(--paper-raised); box-shadow: -0.9rem 0 0 var(--paper-raised), 0.9rem 0 0 var(--paper-raised); }

  .srchead { display: flex; align-items: baseline; gap: .55rem; flex-wrap: wrap; }
  .tag { font-family: var(--font-data); font-size: .74rem; color: var(--seal); }
  .cit { font-weight: 600; font-size: .95rem; }
  .shift {
    margin-left: auto;
    font-family: var(--font-data);
    font-size: .72rem;
    color: var(--ink-soft);
  }
  .shift.up { color: var(--dense); }
  .shift.down { color: var(--lexical); }

  .crumb { margin: .15rem 0 .4rem; font-size: .8rem; color: var(--ink-soft); line-height: 1.45; }

  .channels { display: flex; align-items: center; gap: .5rem; margin-bottom: .45rem; }
  .ch { font-size: .72rem; padding: .05rem .4rem; border-radius: 2px; }
  .ch.dense { color: var(--dense); border: 1px solid var(--dense); }
  .ch.lex { color: var(--lexical); border: 1px solid var(--lexical); }
  .score { margin-left: auto; font-family: var(--font-data); font-size: .72rem; color: var(--ink-soft); }

  .body { margin: 0; font-size: .88rem; line-height: 1.65; color: #2c3138; white-space: pre-wrap; }
  .body.clip { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .more { background: none; border: none; padding: .2rem 0 0; color: var(--ink-soft); font-size: .8rem; cursor: pointer; }
  .more:hover { color: var(--seal); }

  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; gap: 2rem; }
  }
</style>
