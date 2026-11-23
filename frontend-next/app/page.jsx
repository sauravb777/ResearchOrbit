"use client";
import {
  BookOpen,
  Check,
  CircleHelp,
  Download,
  ExternalLink,
  GripVertical,
  History,
  LoaderCircle,
  LockKeyhole,
  Menu,
  MessageCircle,
  MoreHorizontal,
  Orbit,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RefreshCw,
  Rocket,
  Send,
  Upload,
  X,
} from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import "./globals.css";
const API = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api"
).replace(/\/$/, "");
const POLL_INTERVAL_MS = 350;
const POLL_LIMIT = 3000;
const reportTypes = [
  {
    value: "standard_report",
    label: "Standard Report",
    description:
      "A complete, question-driven explanation that connects the most important ideas, methods, findings and unresolved issues.",
  },
  {
    value: "literature_review",
    label: "Literature Review",
    description:
      "A thematic synthesis of what the selected literature says, how the approaches relate and where the findings agree or differ.",
  },
  {
    value: "state_of_art",
    label: "State of the Art / Research Landscape",
    description:
      "A map of the approaches represented in the literature, what they achieve, how they work and what currently limits progress.",
  },
  {
    value: "gap_analysis",
    label: "Research Gap Analysis",
    description:
      "A focused explanation of what is established, what remains unresolved or untested and why those gaps follow from the evidence.",
  },
  {
    value: "comparison",
    label: "Comparative / Methodological Analysis",
    description:
      "A structured explanation and comparison of methods, mechanisms, evidence, trade-offs and conditions where results cannot be compared directly.",
  },
  {
    value: "evidence_review",
    label: "Systematic / Evidence Review",
    description:
      "Assess study designs, consistency, uncertainty and how strongly the evidence supports each conclusion.",
  },
  {
    value: "research_proposal",
    label: "Research Design / Proposal",
    description:
      "Derive a concrete research direction, research questions and evaluation plan from established findings and evidence gaps.",
  },
  {
    value: "paper_analysis",
    label: "Paper / Corpus Analysis",
    description:
      "Explain each paper’s claims, methods, experiments, results and limitations, then connect the findings across papers.",
  },
];
async function apiFetch(path, options = {}, timeoutMs = 30000) {
  const externalSignal = options.signal;
  const controller = externalSignal ? null : new AbortController();
  const timer = controller
    ? window.setTimeout(() => controller.abort(), timeoutMs)
    : null;
  try {
    const response = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        ...(options.headers || {}),
        "X-Workspace-ID": (() => {
          try {
            return localStorage.getItem("researchpilot-workspace-id") || "";
          } catch {
            return "";
          }
        })(),
      },
      signal: externalSignal || controller.signal,
    });
    const raw = await response.text();
    let body = {};
    try {
      body = raw ? JSON.parse(raw) : {};
    } catch {
      body = { detail: raw };
    }
    if (!response.ok)
      throw new Error(
        body.detail || body.message || `Request failed (${response.status})`,
      );
    return body;
  } catch (error) {
    if (error.name === "AbortError") {
      if (externalSignal?.aborted) throw error;
      throw new Error(
        "The request timed out. Check the backend and its terminal logs.",
      );
    }
    throw error;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}
async function pollJob(jobId, onUpdate) {
  let pollFailures = 0;
  for (let attempt = 0; attempt < POLL_LIMIT; attempt += 1) {
    let status;
    try {
      status = await apiFetch(`/jobs/${encodeURIComponent(jobId)}`, {}, 30000);
      pollFailures = 0;
    } catch (error) {
      pollFailures += 1;
      if (pollFailures >= 8) throw error;
      await new Promise((resolve) =>
        window.setTimeout(resolve, POLL_INTERVAL_MS * 2),
      );
      continue;
    }
    onUpdate(status);
    if (status.status === "completed") return status;
    if (status.status === "failed")
      throw new Error(status.error || "Research job failed.");
    if (status.status === "cancelled") {
      const err = new Error("Research job was cancelled.");
      err.code = "CANCELLED";
      throw err;
    }
    if (status.status === "timed_out")
      throw new Error(
        status.error || "Research job exceeded its maximum runtime.",
      );
    await new Promise((resolve) =>
      window.setTimeout(resolve, POLL_INTERVAL_MS),
    );
  }
  throw new Error(
    "The job exceeded the client wait window. The backend may still be running it.",
  );
}
const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
function sourceAuthors(source) {
  return Array.isArray(source?.authors)
    ? source.authors.join(", ")
    : source?.authors || "";
}
function safeExternalUrl(value) {
  if (!value || !/^https?:\/\//i.test(String(value).trim())) return "";
  try {
    const url = new URL(String(value).trim());
    return ["https:", "http:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}
function shortCaseName(name) {
  const value = (name || "Untitled research").trim();
  return value.length > 38 ? `${value.slice(0, 38).trimEnd()}…` : value;
}
function reportTypeLabel(value) {
  return (
    reportTypes.find((x) => x.value === value)?.label ||
    value ||
    "Research Report"
  );
}
function InlineMarkdown({ text }) {
  const tokens = [];
  const pattern =
    /(\[[^\]]+\]\([^\)]+\)|\[S\d+\]|\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*]+\*|_[^_]+_)/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last)
      tokens.push({ type: "text", value: text.slice(last, match.index) });
    const raw = match[0];
    if (raw.startsWith("[")) {
      const m = raw.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
      const href = m ? safeExternalUrl(m[2]) : "";
      tokens.push(
        m && href
          ? { type: "link", label: m[1], href }
          : /^\[S\d+\]$/.test(raw)
            ? { type: "citation", value: raw }
            : { type: "text", value: raw },
      );
    } else if (raw.startsWith("**") || raw.startsWith("__"))
      tokens.push({ type: "strong", value: raw.slice(2, -2) });
    else if (raw.startsWith("`"))
      tokens.push({ type: "code", value: raw.slice(1, -1) });
    else tokens.push({ type: "em", value: raw.slice(1, -1) });
    last = match.index + raw.length;
  }
  if (last < text.length)
    tokens.push({ type: "text", value: text.slice(last) });
  return (
    <>
      {tokens.map((token, i) =>
        token.type === "link" ? (
          <a key={i} href={token.href} target='_blank' rel='noreferrer'>
            {token.label}
          </a>
        ) : token.type === "citation" ? (
          <span key={i} className='citation-tag'>
            {token.value}
          </span>
        ) : token.type === "strong" ? (
          <strong key={i}>{token.value}</strong>
        ) : token.type === "em" ? (
          <em key={i}>{token.value}</em>
        ) : token.type === "code" ? (
          <code key={i}>{token.value}</code>
        ) : (
          <React.Fragment key={i}>{token.value}</React.Fragment>
        ),
      )}
    </>
  );
}
function slugifyHeading(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .slice(0, 70);
}
function extractReportHeadings(markdown) {
  return String(markdown || "")
    .split(/\r?\n/)
    .map((line) => line.match(/^(#{2,3})\s+(.+)$/))
    .filter(Boolean)
    .map((match) => ({
      level: match[1].length,
      title: match[2].replace(/[*_`]/g, "").trim(),
      id: slugifyHeading(match[2]),
    }));
}
function MarkdownReport({ markdown }) {
  const lines = String(markdown || "")
    .replace(/\r\n/g, "\n")
    .split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    if (/^```/.test(line)) {
      const lang = line.slice(3).trim();
      const code = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      i += 1;
      blocks.push(
        <pre className='md-code' key={`code-${i}`}>
          <code data-lang={lang}>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const Tag = `h${heading[1].length}`;
      blocks.push(
        <Tag
          id={heading[1].length >= 2 ? slugifyHeading(heading[2]) : undefined}
          key={`h-${i}`}
        >
          <InlineMarkdown text={heading[2]} />
        </Tag>,
      );
      i += 1;
      continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i]))
        quote.push(lines[i++].replace(/^>\s?/, ""));
      blocks.push(
        <blockquote key={`q-${i}`}>
          <InlineMarkdown text={quote.join(" ")} />
        </blockquote>,
      );
      continue;
    }
    if (
      /^\|/.test(line) &&
      i + 1 < lines.length &&
      /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test(lines[i + 1])
    ) {
      const tableLines = [line];
      i += 2;
      while (i < lines.length && /^\|/.test(lines[i]))
        tableLines.push(lines[i++]);
      const parseRow = (row) =>
        row
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((cell) => cell.trim());
      const head = parseRow(tableLines[0]);
      const rows = tableLines.slice(1).map(parseRow);
      blocks.push(
        <div className='md-table-wrap' key={`table-${i}`}>
          <table>
            <thead>
              <tr>
                {head.map((cell, j) => (
                  <th key={j}>
                    <InlineMarkdown text={cell} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r}>
                  {head.map((_, j) => (
                    <td key={j}>
                      <InlineMarkdown text={row[j] || ""} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*[-*+]\s+/, ""));
      blocks.push(
        <ul key={`ul-${i}`}>
          {items.map((item, j) => (
            <li key={j}>
              <InlineMarkdown text={item} />
            </li>
          ))}
        </ul>,
      );
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
      blocks.push(
        <ol key={`ol-${i}`}>
          {items.map((item, j) => (
            <li key={j}>
              <InlineMarkdown text={item} />
            </li>
          ))}
        </ol>,
      );
      continue;
    }
    const paragraph = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,4})\s|^\s*[-*+]\s+|^\s*\d+[.)]\s+|^>|^```|^\|/.test(lines[i])
    )
      paragraph.push(lines[i++]);
    blocks.push(
      <p key={`p-${i}`}>
        <InlineMarkdown text={paragraph.join(" ")} />
      </p>,
    );
  }
  return <div className='markdown-report'>{blocks}</div>;
}
function SourceDropdown({ references }) {
  return (
    <details className='chat-sources'>
      <summary>
        Where this comes from · {references.length}{" "}
        {references.length === 1 ? "source" : "sources"}
      </summary>
      <div className='chat-sources-list'>
        {references.map((r, i) => (
          <div className='chat-source' key={i}>
            <div className='chat-source-head'>
              {r.tag ? <span className='citation-tag'>[{r.tag}]</span> : null}
              <b>{r.title || "Research report"}</b>
            </div>
            <div className='chat-source-loc'>
              {[r.kind_label, r.location].filter(Boolean).join(" · ")}
            </div>
            {r.excerpt ? <blockquote>{r.excerpt}</blockquote> : null}
          </div>
        ))}
      </div>
    </details>
  );
}
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className='fatal-card'>
        <div className='fatal-icon'>
          <CircleHelp size={22} />
        </div>
        <h2>ResearchOrbit hit an interface error</h2>
        <p>{this.state.error.message}</p>
        <button
          className='secondary-btn'
          onClick={() => window.location.reload()}
        >
          <RefreshCw size={14} /> Reload workspace
        </button>
      </div>
    );
  }
}
function reportEstimateSeconds(paperCount) {
  const estimates = { 1: 4 * 60, 2: 5 * 60, 3: 7 * 60, 4: 8 * 60, 5: 9 * 60 };
  return estimates[Math.min(5, Math.max(1, Number(paperCount) || 1))] || 5 * 60;
}
function ProcessingOverlay({ status, onCancel, clock }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(
      () =>
        setElapsed(
          Math.floor((Date.now() - (clock?.startedAt || Date.now())) / 1000),
        ),
      250,
    );
    return () => window.clearInterval(timer);
  }, [clock?.startedAt]);
  const isDiscovery = clock?.kind === "discovery";
  const duration = isDiscovery
    ? 5.5 * 60
    : reportEstimateSeconds(clock?.paperCount);
  const running = !["complete", "cancelled", "failed", "timed_out"].includes(
    status?.stage,
  );
  const progress =
    running && clock?.startedAt
      ? isDiscovery
        ? Math.min(
            99,
            Math.max(
              0,
              Math.floor(100 * (1 - Math.exp(-elapsed / (duration / 6)))),
            ),
          )
        : Math.min(90, Math.max(0, Math.floor((elapsed / duration) * 90)))
      : 100;
  const reportPhase =
    progress < 60
      ? {
          title: "Deep-reading selected papers",
          message: `Reading and extracting evidence from ${clock?.paperCount || 1} selected ${clock?.paperCount === 1 ? "paper" : "papers"}…`,
        }
      : progress < 80
        ? {
            title: "Analyzing the evidence",
            message:
              "Comparing methods, experiments, findings and limitations across the selected papers…",
          }
        : progress < 90
          ? {
              title: "Synthesizing the research",
              message:
                "Building the report structure and connecting findings across the literature…",
            }
          : {
              title: "Finalizing the report",
              message:
                "Polishing citations, evidence grounding and overall coherence…",
            };
  const displayTitle =
    !isDiscovery && running
      ? reportPhase.title
      : status?.title || "Working on your research";
  const displayMessage =
    !isDiscovery && running
      ? reportPhase.message
      : status?.message || "ResearchOrbit is working through the evidence.";
  const estimateLabel = !isDiscovery
    ? `~${Math.ceil(duration / 60)} min target · runs until complete`
    : "usually 1–2 min · 5.5 min max";
  return (
    <div
      className='processing-overlay'
      role='dialog'
      aria-modal='true'
      aria-live='polite'
    >
      <div className='processing-card'>
        <div className='processing-mark' aria-hidden='true'>
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className='processing-title'>{displayTitle}</div>
        <div className='processing-message'>{displayMessage}</div>
        <div className='processing-progress'>
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className='processing-meta'>
          <span>{progress}%</span>
          <span>
            {Math.floor(elapsed / 60)}m {elapsed % 60}s elapsed
          </span>
          <span>{estimateLabel}</span>
        </div>
        {onCancel && (
          <button className='processing-cancel' onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
function Sidebar({
  open,
  setOpen,
  history,
  result,
  menuRunId,
  setMenuRunId,
  renamingRunId,
  renameText,
  setRenameText,
  beginRename,
  commitRename,
  deleteCase,
  updateCase,
  openHistory,
  newResearch,
}) {
  const [menuPos, setMenuPos] = useState(null);
  const menuRef = useRef(null);
  useEffect(() => {
    if (!menuRunId) return undefined;
    function handleOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target))
        setMenuRunId(null);
    }
    function handleEscape(event) {
      if (event.key === "Escape") setMenuRunId(null);
    }
    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [menuRunId, setMenuRunId]);
  function toggleMenu(event, runId) {
    if (menuRunId === runId) {
      setMenuRunId(null);
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const estimatedMenuHeight = 130;
    const openUpward = rect.bottom + estimatedMenuHeight > window.innerHeight;
    setMenuPos({
      top: openUpward
        ? Math.max(8, rect.top - estimatedMenuHeight)
        : rect.bottom + 4,
      left: Math.max(8, rect.right - 150),
    });
    setMenuRunId(runId);
  }
  const menuItem = menuRunId
    ? history.find((item) => item.run_id === menuRunId)
    : null;
  return (
    <aside className={`sidebar ${open ? "expanded" : "collapsed"}`}>
      <div className='sidebar-header'>
        <button
          className='brand-button'
          onClick={() => !open && setOpen(true)}
          aria-label={open ? "ResearchOrbit home" : "Expand sidebar"}
          title={open ? "ResearchOrbit" : "Expand sidebar"}
        >
          <span className='brand-mark'>
            <Orbit size={17} />
            <span className='brand-expand-mark'>
              <PanelLeftOpen size={15} />
            </span>
          </span>
          {open && <span className='brand-name'>ResearchOrbit</span>}
        </button>
        {open && (
          <button
            className='sidebar-toggle'
            onClick={() => setOpen(false)}
            aria-label='Collapse sidebar'
            title='Collapse sidebar'
          >
            <PanelLeftClose size={17} />
          </button>
        )}
      </div>
      <div className='sidebar-content'>
        <button
          className='new-research'
          onClick={newResearch}
          aria-label='New Research'
        >
          <Plus size={16} />
          <span>New Research</span>
        </button>
        {open && (
          <div className='history-heading'>
            <span>Recent Research</span>
            <History size={14} />
          </div>
        )}
        <nav className='history-list' aria-label='Recent research'>
          {history.map((item) => (
            <div className='history-entry' key={item.run_id}>
              {renamingRunId === item.run_id ? (
                <div className='rename-row'>
                  <input
                    autoFocus
                    maxLength={240}
                    value={renameText}
                    onChange={(e) => setRenameText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(item.run_id);
                      if (e.key === "Escape") setRenameText("");
                    }}
                  />
                  <button
                    onClick={() => commitRename(item.run_id)}
                    aria-label='Save name'
                  >
                    <Check size={13} />
                  </button>
                </div>
              ) : (
                <>
                  <button
                    className={`history-item ${result?.run_id === item.run_id ? "current" : ""}`}
                    onClick={() => openHistory(item.run_id)}
                    title={item.case_name || item.report_title || item.question}
                  >
                    <span>
                      {shortCaseName(
                        item.case_name || item.report_title || item.question,
                      )}
                    </span>
                    {open && (
                      <small>
                        {item.version_count ? `${item.version_count} v` : "1 v"}
                      </small>
                    )}
                  </button>
                  {open && (
                    <button
                      className='history-menu'
                      onClick={(e) => toggleMenu(e, item.run_id)}
                      aria-label='Research actions'
                      aria-haspopup='true'
                      aria-expanded={menuRunId === item.run_id}
                    >
                      <MoreHorizontal size={15} />
                    </button>
                  )}
                </>
              )}
            </div>
          ))}
        </nav>
      </div>
      {open && menuItem && menuPos && (
        <div
          ref={menuRef}
          className='case-menu'
          style={{ top: `${menuPos.top}px`, left: `${menuPos.left}px` }}
          role='menu'
        >
          <button
            role='menuitem'
            onClick={() => {
              beginRename(menuItem);
              setMenuRunId(null);
            }}
          >
            Rename
          </button>
          {menuItem.pinned && (
            <button
              role='menuitem'
              onClick={() => {
                updateCase(menuItem.run_id, { pinned: false });
                setMenuRunId(null);
              }}
            >
              Unpin
            </button>
          )}
          <button
            role='menuitem'
            className='danger'
            onClick={() => {
              deleteCase(menuItem.run_id);
              setMenuRunId(null);
            }}
          >
            Delete research trace
          </button>
        </div>
      )}
    </aside>
  );
}
function Setup({
  question,
  setQuestion,
  title,
  setTitle,
  reportType,
  setReportType,
  start,
  loading,
}) {
  return (
    <div className='setup-page'>
      <div className='setup-hero'>
        <span className='eyebrow'>RESEARCH WORKSPACE</span>
        <h1>What do you want to understand?</h1>
        <p>
          ResearchOrbit searches arXiv, builds a relevance-ranked corpus,
          verifies full-text access for selected papers and builds a report
          around the analytical purpose you choose.
        </p>
      </div>
      <div className='question-card'>
        <label htmlFor='research-question'>Research question</label>
        <textarea
          id='research-question'
          rows={4}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder='Ask a focused research question…'
          disabled={loading}
        />
        <div className='question-footer'>
          <span>{question.trim().length}/12000</span>
          <button
            className='primary-btn'
            onClick={start}
            disabled={!question.trim() || !title.trim() || loading}
          >
            <Rocket size={15} /> Find &amp; Research
          </button>
        </div>
      </div>
      <div className='title-row'>
        <label htmlFor='report-title'>
          Report title <span>Required</span>
        </label>
        <input
          id='report-title'
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder='Enter the title for your research report'
          maxLength={240}
          disabled={loading}
          required
        />
        <small className='title-hint'>
          You choose the title. ResearchOrbit will use it exactly as entered.
        </small>
      </div>
      <section className='report-type-picker'>
        <div className='picker-heading'>
          <div>
            <span className='eyebrow'>ANALYTICAL MODE</span>
            <h2>Choose how you want the research understood</h2>
          </div>
          <span>
            One full research-reading regime · different analytical blueprints
          </span>
        </div>
        <div className='report-type-grid'>
          {reportTypes.map((type, index) => (
            <button
              key={type.value}
              type='button'
              className={`report-type-card ${reportType === type.value ? "selected" : ""}`}
              onClick={() => setReportType(type.value)}
              aria-pressed={reportType === type.value}
            >
              <span className='report-type-number'>0{index + 1}</span>
              <span className='report-type-label'>{type.label}</span>
              <span className='report-type-description'>
                {type.description}
              </span>
              <span className='report-type-check'>
                {reportType === type.value ? "Selected" : "Select"}
              </span>
            </button>
          ))}
        </div>
      </section>
      <div className='setup-note'>
        <span className='state-chip selected'>REVIEW FIRST</span>
        <strong>Top 15, then choose what to read</strong>
        <span>
          ResearchOrbit searches arXiv for a ranked evidence corpus. It
          recommends papers for deep reading using relevance and evidence
          coverage rather than a fixed ranking position. You can adjust the
          selection before reading. Each selected paper is deeply analyzed and
          synthesized into one detailed report.
        </span>
      </div>
    </div>
  );
}
function Screening({
  candidates,
  loading,
  jobStatus,
  generate,
  back,
  onToggle,
  uploadedCandidates,
  onUpload,
  uploadBusy,
}) {
  const red = candidates.filter((s) => s.role === "deep_read").length;
  const green = candidates.filter((s) => s.role !== "deep_read").length;
  const ranked = [...candidates].sort(
    (a, b) => Number(b.score || 0) - Number(a.score || 0),
  );
  const maxRawRelevance = Math.max(
    0,
    ...ranked.map((s) => Number(s.relevance_score ?? s.score ?? 0)),
  );
  const relativeRelevance = (source) => {
    const raw = Number(source.relevance_score ?? source.score ?? 0);
    if (!Number.isFinite(raw) || raw <= 0 || maxRawRelevance <= 0) return 0;
    return Math.round((raw / maxRawRelevance) * 84);
  };
  return (
    <div className='screening-page'>
      <div className='screening-head'>
        <div>
          <span className='eyebrow'>RESEARCH CORPUS</span>
          <h1>Choose what ResearchOrbit should read</h1>
          <p>
            ResearchOrbit searches arXiv for up to 15 strong papers and ranks
            them by relevance. Review the ranked corpus and select the papers
            you want to study in depth. Only selected papers enter the
            deep-reading and report pipeline.
          </p>
        </div>
        <div className='screening-stats'>
          <div className='corpus-stat red-stat'>
            <b>{red}</b>
            <span>deep read</span>
          </div>
          <div className='corpus-stat green-stat'>
            <b>{green}</b>
            <span>no read</span>
          </div>
          <div className='corpus-stat'>
            <b>{candidates.length}</b>
            <span>candidates</span>
          </div>
        </div>
      </div>
      {loading && (
        <div className='inline-progress'>
          <LoaderCircle size={15} className='spin' />{" "}
          {jobStatus?.message || "Building the evidence corpus…"}{" "}
          <span>{jobStatus?.progress || 0}%</span>
        </div>
      )}
      {!loading && candidates.length === 0 && (
        <div className='empty-state'>
          <SearchIcon />
          <h2>No candidate papers were found</h2>
          <p>arXiv returned no usable literature for this question.</p>
          <button className='secondary-btn' onClick={back}>
            Back to research
          </button>
        </div>
      )}
      {!loading && candidates.length > 0 && (
        <>
          <section className='paper-section'>
            <div className='section-title'>
              <div>
                <span className='eyebrow'>TOP ARXIV CORPUS</span>
                <h2>Choose papers to deep read</h2>
              </div>
              <span>
                {red} / 5 selected · {candidates.length} found
              </span>
            </div>
            <div className='candidate-list'>
              {ranked.map((source, index) => {
                const isRed = source.role === "deep_read";
                return (
                  <article
                    className={`candidate-card ${isRed ? "red-card" : "green-card"}`}
                    key={source.id || index}
                  >
                    <div
                      className={`candidate-state-bar ${isRed ? "red" : "green"}`}
                      aria-hidden='true'
                    />
                    <div className='candidate-index'>
                      {String(index + 1).padStart(2, "0")}
                    </div>
                    <div className='candidate-content'>
                      <div className='candidate-topline'>
                        <div className='candidate-title-wrap'>
                          <span className='rank-label'>RANK {index + 1}</span>
                          <h3>{source.title || "Untitled paper"}</h3>
                          <div className='paper-meta'>
                            {sourceAuthors(source)}{" "}
                            {source.year && `· ${source.year}`}{" "}
                            {source.venue && `· ${source.venue}`}
                          </div>
                        </div>
                        <div className='candidate-actions'>
                          {source.relevance_score != null && (
                            <span
                              className='relevance-inline'
                              title='Rank is the canonical raw relevance ordering. Automatic selection uses raw >=39% eligibility; the 44% threshold affects corpus size, not rank order.'
                            >
                              Relative relevance {relativeRelevance(source)}%
                            </span>
                          )}
                          <div
                            className='selection-buttons'
                            role='radiogroup'
                            aria-label={`Reading selection for ${source.title || "paper"}`}
                          >
                            <button
                              type='button'
                              className={`selection-choice selected-choice ${isRed ? "active" : ""}`}
                              onClick={() =>
                                !isRed && red < 5 && onToggle(source.id)
                              }
                              role='radio'
                              aria-checked={isRed}
                              disabled={!isRed && red >= 5}
                            >
                              Deep read
                            </button>
                            <button
                              type='button'
                              className={`selection-choice unselected-choice ${!isRed ? "active" : ""}`}
                              onClick={() => isRed && onToggle(source.id)}
                              role='radio'
                              aria-checked={!isRed}
                            >
                              Skip
                            </button>
                          </div>
                        </div>
                      </div>
                      <p>{source.abstract || "No abstract available."}</p>
                      <div className='candidate-bottom'>
                        <div className='candidate-reasons'>
                          {(source.relevance_reasons || [])
                            .slice(0, 4)
                            .map((reason, i) => (
                              <span key={i}>{reason}</span>
                            ))}
                          {(source.missing_ranking_signals || []).length >
                            0 && (
                            <span className='metadata-note'>
                              Some ranking metadata unavailable, so it was
                              omitted from the score.
                            </span>
                          )}
                        </div>
                        <div className='candidate-links'>
                          {safeExternalUrl(source.url) && (
                            <a
                              href={safeExternalUrl(source.url)}
                              target='_blank'
                              rel='noopener noreferrer'
                            >
                              <ExternalLink size={12} /> arXiv
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
          <section className='paper-section uploaded-section'>
            <div className='section-title'>
              <div>
                <span className='eyebrow'>OPTIONAL INPUT</span>
                <h2>Bring your own papers</h2>
              </div>
              <button
                className='secondary-btn'
                onClick={onUpload}
                disabled={uploadBusy}
              >
                <Upload size={14} /> Add papers
              </button>
            </div>
            <div className='upload-empty'>
              <Upload size={17} />
              <span>
                Uploaded papers use the same relevance pipeline. You can select
                or deselect any uploaded paper before generating the report.
              </span>
            </div>
          </section>
          <div className='screening-footer'>
            <button className='secondary-btn' onClick={back}>
              Back to question
            </button>
            <button className='primary-btn' onClick={generate} disabled={!red}>
              <Orbit size={15} /> Deep Read {red} Paper{red === 1 ? "" : "s"}{" "}
              &amp; Generate Report
            </button>
          </div>
        </>
      )}
    </div>
  );
}
function SearchIcon() {
  return (
    <div className='empty-state-icon'>
      <BookOpen size={18} />
    </div>
  );
}
function Report({
  result,
  reportRef,
  onSelection,
  selectedText,
  explain,
  reanalyze,
  uploading,
  download,
  downloadingKind,
  versions,
  versionsOpen,
  setVersionsOpen,
  loadVersion,
}) {
  const hasReport = Boolean(result.report_markdown?.trim());
  const fileInputRef = useRef(null);
  const reportHeadings = extractReportHeadings(result.report_markdown);
  const showToc = String(result.report_markdown || "").length > 7000;
  return (
    <div className='report-page'>
      <div className='report-head'>
        <div>
          <span className='eyebrow'>
            {reportTypeLabel(result.report_type)} ·{" "}
            {result.sources?.length || 0} papers · VERSION{" "}
            {result.version || 1}{" "}
          </span>
          <h1>{result.report_title || "Research Report"}</h1>
          <p>{result.question}</p>
        </div>
        <div className='report-actions'>
          <button
            className='secondary-btn'
            onClick={() => setVersionsOpen(!versionsOpen)}
          >
            <History size={14} /> Versions
          </button>
          <button
            type='button'
            className='secondary-btn upload-btn'
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            <Plus size={14} /> Add up to 2 papers
          </button>
          <input
            ref={fileInputRef}
            className='upload-input-hidden'
            type='file'
            accept='.pdf,application/pdf'
            multiple
            onChange={reanalyze}
          />
          <div className='download-actions'>
            {[
              ["pdf", "PDF"],
              ["docx", "DOCX"],
              ["latex", "LaTeX"],
              ["bib", "BIB"],
            ].map(([kind, label]) => (
              <button
                key={kind}
                onClick={() => download(kind)}
                disabled={
                  Boolean(downloadingKind) ||
                  (kind === "bib" ? !result.sources?.length : !hasReport)
                }
                aria-busy={downloadingKind === kind}
              >
                {downloadingKind === kind ? (
                  <>
                    <LoaderCircle size={13} className='spin' /> Preparing…
                  </>
                ) : (
                  <>
                    <Download size={13} /> {label}
                  </>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
      {versionsOpen && (
        <div className='version-drawer'>
          <b>Report history</b>
          <span>Immutable versions of this research case</span>
          <div className='version-list'>
            {versions.map((v) => (
              <button
                key={v.run_id}
                className={v.run_id === result.run_id ? "current" : ""}
                onClick={() => loadVersion(v.run_id)}
              >
                <span>
                  v{v.version} · {v.report_title || "Research Report"}
                </span>
                <small>
                  {v.sources_selected || 0} papers · {v.papers_deep_read || 0}{" "}
                  deep reads
                </small>
              </button>
            ))}
          </div>
        </div>
      )}
      <section className='section-card report-reader'>
        <div className='section-head'>
          <div>
            <span className='eyebrow'>REPORT</span>
            <h2>Research report</h2>
            <p>Select any passage and ask the chat to explain it.</p>
          </div>
          <button
            className='secondary-btn'
            onClick={explain}
            disabled={!selectedText}
          >
            <MessageCircle size={13} /> Explain selection
          </button>
        </div>
        <article
          ref={reportRef}
          onMouseUp={() => {
            const text = window.getSelection()?.toString().trim() || "";
            onSelection(text);
          }}
          className='report-surface'
        >
          {hasReport ? (
            <MarkdownReport
              markdown={String(result.report_markdown).replace(
                /^# [^\n]+\n?/,
                "",
              )}
            />
          ) : (
            <div className='empty-report'>No report content available.</div>
          )}
        </article>
      </section>
    </div>
  );
}
function ChatPanel({
  result,
  chat,
  chatText,
  setChatText,
  chatLoading,
  ask,
  chatMode,
  setChatMode,
  selectedText,
  explainSelection,
  mobileOpen,
  setMobileOpen,
  onResize,
  stopChat,
}) {
  const scrollRef = useRef(null);
  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [chat.length, chatLoading]);
  const unlocked = Boolean(result?.run_id && result?.report_markdown);
  const suggestions =
    chatMode === "verify"
      ? [
          "Verify this claim against the selected papers.",
          "Does paper S1 actually support this statement?",
          "Which paper supports or contradicts this claim?",
        ]
      : [
          "Explain the main mechanism in simple terms.",
          "Explain paper S2 in two paragraphs.",
          "What is the strongest result, and what does it mean?",
          "What important gap remains?",
        ];
  return (
    <aside
      className={`chat-panel ${mobileOpen ? "mobile-open" : "mobile-closed"}`}
      aria-label='Research chat'
    >
      <button
        className='chat-resize-handle'
        onMouseDown={onResize}
        aria-label='Resize research chat'
        title='Drag to resize'
      >
        <GripVertical size={15} />
      </button>
      <div className='chat-head'>
        <div>
          <span className='eyebrow'>GROUNDED IN YOUR CORPUS</span>
          <strong>Research Chat</strong>
        </div>
        <button
          className='chat-mobile-toggle'
          onClick={() => setMobileOpen(false)}
          aria-label='Minimize chat'
        >
          <X size={16} />
        </button>
      </div>
      {!unlocked ? (
        <div className='chat-lock-card chat-locked'>
          <div className='lock-wrap'>
            <LockKeyhole size={22} />
          </div>
          <span>LOCKED UNTIL REPORT</span>
          <h3>Research chat unlocks when your report is ready.</h3>
          <p>
            After the report is generated, chat can answer questions from the
            report and the selected deep-read papers.
          </p>
        </div>
      ) : (
        <>
          <div
            className='chat-mode-switch'
            role='tablist'
            aria-label='Research chat mode'
          >
            <button
              type='button'
              className={chatMode === "ask" ? "active" : ""}
              onClick={() => setChatMode("ask")}
              role='tab'
              aria-selected={chatMode === "ask"}
            >
              Chat
            </button>
            <button
              type='button'
              className={chatMode === "verify" ? "active" : ""}
              onClick={() => setChatMode("verify")}
              role='tab'
              aria-selected={chatMode === "verify"}
            >
              Verify claims
            </button>
          </div>
          <div className='chat-mode-note'>
            {chatMode === "verify"
              ? "Verify a specific claim against the research papers. Verification evidence appears only for verification requests."
              : "Ask normally about the report or papers. Verification is off in normal Chat mode."}
          </div>
          {chat.length === 0 && (
            <div className='chat-suggestions'>
              {suggestions.map((item, i) => (
                <button
                  key={i}
                  type='button'
                  onClick={() => ask(item)}
                  disabled={chatLoading}
                >
                  {item}
                </button>
              ))}
            </div>
          )}
          {selectedText && (
            <div className='selection-context'>
              <div>
                <span>Selected report text</span>
                <p>
                  {selectedText.length > 220
                    ? `${selectedText.slice(0, 220)}…`
                    : selectedText}
                </p>
              </div>
              <button
                onClick={chatMode === "verify" ? ask : explainSelection}
                disabled={chatLoading}
              >
                <Orbit size={13} />{" "}
                {chatMode === "verify" ? "Verify selection" : "Explain"}
              </button>
            </div>
          )}
          <div className='chat-messages' ref={scrollRef}>
            {chat.length === 0 && (
              <div className='chat-empty'>
                <div className='chat-empty-mark'>
                  <MessageCircle size={18} />
                </div>
                <h3>
                  {chatMode === "verify"
                    ? "Verify a claim"
                    : "Ask about the research"}
                </h3>
                <p>
                  {chatMode === "verify"
                    ? "Paste a claim or ask whether a statement is supported by the papers."
                    : "Ask about any paper, method, experiment, result, limitation, section or claim."}
                </p>
              </div>
            )}
            {chat.map((message, index) => (
              <div key={index} className={`chat-message-row ${message.role}`}>
                <div
                  className={`chat-message ${message.role} ${message.error ? "error" : ""}`}
                >
                  <span className='message-mode'>
                    {message.mode ||
                      (message.role === "user" ? "Chat" : "ResearchOrbit")}
                  </span>
                  <div>
                    {message.role === "assistant" && !message.error ? (
                      <InlineMarkdown text={message.content} />
                    ) : (
                      message.content
                    )}
                  </div>
                  {message.needsClarification &&
                    message.clarificationOptions?.length > 0 && (
                      <div className='clarification-options'>
                        {message.clarificationOptions.map(
                          (option, optionIndex) => (
                            <button
                              key={optionIndex}
                              type='button'
                              onClick={() => ask(option)}
                              disabled={chatLoading}
                            >
                              {option}
                            </button>
                          ),
                        )}
                      </div>
                    )}
                  {message.role === "assistant" &&
                    message.references?.length > 0 && (
                      <SourceDropdown references={message.references} />
                    )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className='chat-message-row assistant'>
                <div className='chat-message assistant typing'>
                  <span className='typing-dots'>
                    <i />
                    <i />
                    <i />
                  </span>
                  ResearchOrbit is thinking…
                </div>
              </div>
            )}
          </div>
          <form
            className='chat-composer'
            onSubmit={(e) => {
              e.preventDefault();
              if (chatText.trim()) ask(chatText.trim());
            }}
          >
            <input
              value={chatText}
              onChange={(e) => setChatText(e.target.value)}
              placeholder={
                chatMode === "verify"
                  ? "Enter a claim to verify…"
                  : "Ask a research question…"
              }
              aria-label={
                chatMode === "verify" ? "Claim to verify" : "Research question"
              }
            />
            {chatLoading ? (
              <button
                type='button'
                className='chat-stop'
                onMouseDown={(e) => e.preventDefault()}
                onClick={stopChat}
                aria-label='Stop response'
                title='Stop response'
              >
                <span />
              </button>
            ) : (
              <button
                type='submit'
                disabled={!chatText.trim()}
                aria-label='Send'
              >
                <Send size={15} />
              </button>
            )}
          </form>
        </>
      )}
    </aside>
  );
}
function App() {
  const [leftOpen, setLeftOpen] = useState(true);
  const [mobileChatOpen, setMobileChatOpen] = useState(false);
  const [chatWidth, setChatWidth] = useState(390);
  const chatWidthLoaded = useRef(false);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("researchpilot-chat-width");
      const n = raw === null ? NaN : Number(raw);
      if (Number.isFinite(n)) setChatWidth(Math.min(520, Math.max(320, n)));
    } catch (e) {
    }
    chatWidthLoaded.current = true;
  }, []);
  const resizingChat = useRef(false);
  const [question, setQuestion] = useState("");
  const [title, setTitle] = useState("");
  const [reportType, setReportType] = useState("standard_report");
  const [stage, setStage] = useState("setup");
  const [loading, setLoading] = useState(false);
  const [downloadingKind, setDownloadingKind] = useState("");
  const [jobStatus, setJobStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [versions, setVersions] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [uploadedCandidates, setUploadedCandidates] = useState([]);
  const [discoveryJobId, setDiscoveryJobId] = useState(null);
  const [selectionOverridden, setSelectionOverridden] = useState(false);
  const [chat, setChat] = useState([]);
  const [chatText, setChatText] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMode, setChatMode] = useState("ask");
  const [selectedText, setSelectedText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [activeJobId, setActiveJobId] = useState(null);
  const [activeJobKind, setActiveJobKind] = useState(null);
  const [progressClock, setProgressClock] = useState(null);
  const chatAbortRef = useRef(null);
  const chatPendingQuestionRef = useRef("");
  const chatRequestIdRef = useRef(null);
  const chatRequestSeqRef = useRef(0);
  const [menuRunId, setMenuRunId] = useState(null);
  const [renamingRunId, setRenamingRunId] = useState(null);
  const [renameText, setRenameText] = useState("");
  const [versionsOpen, setVersionsOpen] = useState(false);
  const reportRef = useRef(null);
  const uploadRef = useRef(null);
  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await apiFetch("/history", {}, 20000));
    } catch {
    }
  }, []);
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);
  useEffect(() => {
    if (!chatWidthLoaded.current) return;
    try {
      localStorage.setItem("researchpilot-chat-width", String(chatWidth));
    } catch (e) {
    }
  }, [chatWidth]);
  useEffect(() => {
    const move = (event) => {
      if (!resizingChat.current) return;
      setChatWidth(
        Math.min(520, Math.max(320, window.innerWidth - event.clientX)),
      );
    };
    const up = () => {
      resizingChat.current = false;
      document.body.classList.remove("resizing-chat");
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, []);
  const reset = useCallback(() => {
    setQuestion("");
    setTitle("");
    setResult(null);
    setChat([]);
    setSelectedText("");
    setCandidates([]);
    setUploadedCandidates([]);
    setDiscoveryJobId(null);
    setSelectionOverridden(false);
    setJobStatus(null);
    setActiveJobId(null);
    setActiveJobKind(null);
    setProgressClock(null);
    setStage("setup");
    setError("");
    setVersions([]);
    setVersionsOpen(false);
    setMobileChatOpen(true);
  }, []);
  function hydrateChat(data) {
    const turns = Array.isArray(data?.conversation) ? data.conversation : [];
    return turns.flatMap((turn) => [
      { role: "user", content: turn.question, mode: turn.mode },
      {
        role: "assistant",
        references: turn.references || [],
        content: turn.answer || "",
        mode: turn.needs_clarification
          ? "ResearchOrbit · Clarification"
          : turn.mode,
        needsClarification: Boolean(turn.needs_clarification),
        clarificationOptions: turn.clarification_options || [],
      },
    ]);
  }
  async function discoverCandidates() {
    const clean = question.trim();
    const cleanTitle = title.trim();
    if (!clean || !cleanTitle || loading) {
      if (!cleanTitle)
        setError("Please enter a report title before starting the research.");
      return;
    }
    setLoading(true);
    setError("");
    setStage("screening");
    setJobStatus({
      title: "Finding relevant papers",
      message: "Searching and ranking the literature…",
      progress: 0,
    });
    setProgressClock({ kind: "discovery", startedAt: Date.now() });
    setCandidates([]);
    setActiveJobKind("discovery");
    try {
      const job = await apiFetch(
        "/research/discover",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: clean,
            title: title.trim(),
            report_type: reportType,
          }),
        },
        30000,
      );
      setDiscoveryJobId(job.job_id);
      setActiveJobId(job.job_id);
      const done = await pollJob(job.job_id, (status) =>
        setJobStatus({
          ...status,
          title:
            status.stage === "screening"
              ? "Ranking relevant papers"
              : "Finding relevant papers",
        }),
      );
      setJobStatus({
        title: "Finished",
        message: "Candidate papers are ready for review.",
        progress: 100,
      });
      await wait(80);
      const found = done.result?.candidates || [];
      setCandidates(found);
      setActiveJobId(null);
      setActiveJobKind(null);
      setProgressClock(null);
      setLoading(false);
      setJobStatus(null);
    } catch (e) {
      if (e.code !== "CANCELLED")
        setError(e.message || "Candidate discovery failed.");
      setStage("setup");
      setDiscoveryJobId(null);
      setActiveJobId(null);
      setActiveJobKind(null);
      setProgressClock(null);
      setLoading(false);
      setJobStatus(null);
    }
  }
  function toggleCandidate(sourceId) {
    setSelectionOverridden(true);
    setCandidates((current) =>
      current.map((source) =>
        source.id === sourceId
          ? {
              ...source,
              role: source.role === "deep_read" ? "no_read" : "deep_read",
            }
          : source,
      ),
    );
    setUploadedCandidates((current) =>
      current.map((source) =>
        source.id === sourceId
          ? {
              ...source,
              role: source.role === "deep_read" ? "no_read" : "deep_read",
            }
          : source,
      ),
    );
  }
  async function generateReport() {
    if (!discoveryJobId || loading) return;
    const redIds = candidates
      .filter((source) => source.role === "deep_read")
      .map((source) => source.id);
    if (!redIds.length) {
      setError("Select at least one paper for Deep Read.");
      return;
    }
    if (redIds.length > 5) {
      setError(
        "Deep Read supports a maximum of 5 papers. Remove a selection before generating the report.",
      );
      return;
    }
    setLoading(true);
    setError("");
    setProgressClock({
      kind: "report",
      startedAt: Date.now(),
      paperCount: redIds.length,
    });
    setActiveJobKind("report");
    setJobStatus({
      title: "Building your research report",
      message:
        "Deep-reading the selected evidence corpus and synthesizing the literature…",
      progress: 0,
    });
    try {
      const effectiveTitle = title.trim();
      if (!effectiveTitle) throw new Error("A report title is required.");
      const job = await apiFetch(
        "/research/run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            title: effectiveTitle,
            report_type: reportType,
            discovery_job_id: discoveryJobId,
            selected_source_ids: redIds,
            selection_overridden: true,
            extra_sources: uploadedCandidates.map((item) => ({ ...item })),
          }),
        },
        30000,
      );
      setActiveJobId(job.job_id);
      const done = await pollJob(job.job_id, (status) =>
        setJobStatus({
          ...status,
          title:
            status.stage === "synthesis"
              ? "Finalizing report"
              : "Researching evidence",
          message: status.message || "Processing your research…",
        }),
      );
      const next = done.result;
      if (!next?.run_id)
        throw new Error(
          "The backend did not return a completed research result.",
        );
      setJobStatus({
        title: "Finished",
        message: "Your evidence-grounded report is ready.",
        progress: 100,
      });
      await wait(80);
      setResult(next);
      setQuestion(next.question || question);
      setTitle(next.report_title || effectiveTitle);
      setChat(hydrateChat(next));
      setStage("report");
      setActiveJobId(null);
      setActiveJobKind(null);
      setProgressClock(null);
      setLoading(false);
      setJobStatus(null);
      void refreshHistory();
    } catch (e) {
      if (e.code !== "CANCELLED")
        setError(e.message || "Research generation failed.");
      setLoading(false);
      setActiveJobId(null);
      setActiveJobKind(null);
      setProgressClock(null);
      setJobStatus(null);
      setStage("screening");
    }
  }
  async function cancelActiveJob() {
    if (!activeJobId) return;
    const jobId = activeJobId;
    const kind = activeJobKind;
    setActiveJobId(null);
    setActiveJobKind(null);
    setProgressClock(null);
    setLoading(false);
    setUploading(false);
    setJobStatus(null);
    if (kind === "discovery") {
      setDiscoveryJobId(null);
      setCandidates([]);
      setStage("setup");
      setError("Paper discovery cancelled.");
    } else {
      setStage("screening");
      setError(
        "Report generation cancelled. Your research corpus is still available.",
      );
    }
    try {
      await apiFetch(
        `/jobs/${encodeURIComponent(jobId)}/cancel`,
        { method: "POST" },
        10000,
      );
    } catch (e) {
      setError(
        e.message ||
          "The job was stopped locally, but the backend could not confirm cancellation.",
      );
    }
  }
  async function ask(text = chatText) {
    const q = text.trim();
    if (!result?.run_id || !q || chatLoading) return;
    const context = selectedText
      ? `\n\nSelected report context:\n${selectedText}`
      : "";
    const controller = new AbortController();
    const requestId = globalThis.crypto?.randomUUID
      ? globalThis.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const requestSeq = ++chatRequestSeqRef.current;
    chatAbortRef.current = controller;
    chatRequestIdRef.current = requestId;
    chatPendingQuestionRef.current = q;
    setChat((c) => [
      ...c,
      {
        role: "user",
        content: q + context,
        mode: chatMode === "verify" ? "Verify" : "Chat",
      },
    ]);
    setChatText("");
    setChatLoading(true);
    let chatTimer = null;
    try {
      chatTimer = window.setTimeout(() => controller.abort(), 45000);
      const data = await apiFetch(
        `/research/${encodeURIComponent(result.run_id)}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: result.run_id,
            request_id: requestId,
            question: q + context,
            mode: chatMode === "verify" ? "Verify" : "Ask",
          }),
          signal: controller.signal,
        },
        45000,
      );
      if (requestSeq !== chatRequestSeqRef.current || controller.signal.aborted)
        return;
      const answer =
        data.answer || data.reasoning || data.content || JSON.stringify(data);
      setChat((c) => [
        ...c,
        {
          role: "assistant",
          content: answer,
          mode: data.needs_clarification
            ? "ResearchOrbit · Clarification"
            : chatMode === "verify"
              ? `Verification${data.verdict ? ` · ${String(data.verdict).replaceAll("_", " ")}` : ""}`
              : "ResearchOrbit",
          references: chatMode === "verify" ? data.references || [] : [],
          needsClarification: Boolean(data.needs_clarification),
          clarificationOptions: data.clarification_options || [],
        },
      ]);
      setResult((current) =>
        current
          ? {
              ...current,
              conversation: [
                ...(current.conversation || []),
                {
                  mode: chatMode === "verify" ? "verify" : "ask",
                  question: q + context,
                  answer,
                  evidence_ids: data.evidence_ids || [],
                  source_ids: data.source_ids || [],
                  references:
                    chatMode === "verify" ? data.references || [] : [],
                  needs_clarification: Boolean(data.needs_clarification),
                  clarification_options: data.clarification_options || [],
                },
              ],
            }
          : current,
      );
      setSelectedText("");
    } catch (e) {
      if (e.name !== "AbortError" && requestSeq === chatRequestSeqRef.current)
        setChat((c) => [
          ...c,
          {
            role: "assistant",
            content: `Unable to answer right now: ${e.message}`,
            error: true,
          },
        ]);
    } finally {
      if (chatTimer) window.clearTimeout(chatTimer);
      if (chatAbortRef.current === controller) chatAbortRef.current = null;
      if (chatRequestIdRef.current === requestId)
        chatRequestIdRef.current = null;
      if (requestSeq === chatRequestSeqRef.current) {
        chatPendingQuestionRef.current = "";
        setChatLoading(false);
      }
    }
  }
  function stopChat() {
    const requestId = chatRequestIdRef.current;
    const runId = result?.run_id;
    chatRequestSeqRef.current += 1;
    if (chatAbortRef.current) chatAbortRef.current.abort();
    if (runId && requestId) {
      void apiFetch(
        `/research/${encodeURIComponent(runId)}/chat/cancel?request_id=${encodeURIComponent(requestId)}`,
        { method: "POST" },
        10000,
      ).catch(() => {});
    }
    chatAbortRef.current = null;
    chatRequestIdRef.current = null;
    chatPendingQuestionRef.current = "";
    setChat((c) => [
      ...c,
      {
        role: "assistant",
        content: "Response stopped.",
        mode: "ResearchOrbit",
      },
    ]);
    setChatLoading(false);
  }
  async function explainSelection() {
    if (!result?.run_id || !selectedText || chatLoading) return;
    const context = selectedText;
    const q =
      "Verify this selected passage against the research papers and explain what the papers actually support.";
    setChat((c) => [
      ...c,
      {
        role: "user",
        content: `${q}\n\nSelected passage:\n${context}`,
        mode: "Ask",
      },
    ]);
    setSelectedText("");
    setChatLoading(true);
    try {
      const data = await apiFetch(
        `/research/${encodeURIComponent(result.run_id)}/explain-selection`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: result.run_id,
            selected_text: context,
            question: q,
            mode: "Verify",
          }),
        },
        120000,
      );
      setChat((c) => [
        ...c,
        {
          role: "assistant",
          content: data.answer || data.content || JSON.stringify(data),
          mode: "Ask",
          references: chatMode === "verify" ? data.references || [] : [],
        },
      ]);
    } catch (e) {
      setChat((c) => [
        ...c,
        {
          role: "assistant",
          content: `Selection explanation failed: ${e.message}`,
          error: true,
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }
  async function handleUpload(event) {
    const files = [...(event.target.files || [])];
    event.target.value = "";
    if (!files.length || uploading) return;
    if (files.length > 16) {
      setError("You can upload at most 16 PDFs in one batch.");
      return;
    }
    setUploading(true);
    setError("");
    setJobStatus({
      title: "Analyzing uploaded papers",
      message: "Reading and ranking your uploaded papers…",
      progress: 10,
    });
    try {
      if (stage === "report") {
        if (files.length > 2)
          throw new Error(
            "You can add at most 2 papers after the report is generated.",
          );
        const formData = new FormData();
        files.forEach((file) => formData.append("files", file));
        const job = await apiFetch(
          `/research/${encodeURIComponent(result.run_id)}/reanalyze/job`,
          { method: "POST", body: formData },
          30000,
        );
        setActiveJobId(job.job_id);
        const done = await pollJob(job.job_id, (status) =>
          setJobStatus({
            ...status,
            title:
              status.stage === "deep_analysis"
                ? "Deep analyzing added papers"
                : "Updating your report",
            message:
              status.message || "Updating the research corpus and report…",
          }),
        );
        setJobStatus({
          title: "Finished",
          message: "The corpus and report have been updated.",
          progress: 100,
        });
        await wait(80);
        const updated = done.result;
        if (!updated?.run_id)
          throw new Error(
            "The backend did not return the updated research result.",
          );
        setResult(updated);
        setChat(hydrateChat(updated));
        await refreshHistory();
        await loadVersions(updated.run_id);
        setActiveJobId(null);
      } else if (stage === "screening" && discoveryJobId) {
        const formData = new FormData();
        files.forEach((file) => formData.append("files", file));
        const job = await apiFetch(
          `/research/discover/upload/job?discovery_job_id=${encodeURIComponent(discoveryJobId)}`,
          { method: "POST", body: formData },
          30000,
        );
        setActiveJobId(job.job_id);
        const done = await pollJob(job.job_id, (status) =>
          setJobStatus({
            ...status,
            title:
              status.stage === "upload_ranking"
                ? "Ranking uploaded papers"
                : "Analyzing uploaded papers",
            message:
              status.message || "Reading and ranking your uploaded papers…",
          }),
        );
        setJobStatus({
          title: "Finished",
          message: "Uploaded papers are ranked and ready for review.",
          progress: 100,
        });
        await wait(80);
        const uploaded = (done.result?.candidates || []).map((item) => ({
          ...item,
        }));
        setUploadedCandidates((previous) => [
          ...previous.filter(
            (old) => !uploaded.some((item) => item.id === old.id),
          ),
          ...uploaded,
        ]);
        setCandidates((previous) => [
          ...previous.filter(
            (old) => !uploaded.some((item) => item.id === old.id),
          ),
          ...uploaded,
        ]);
        setActiveJobId(null);
      } else {
        throw new Error("Start a research question before uploading papers.");
      }
    } catch (e) {
      setError(e.message || "Paper processing failed.");
      setActiveJobId(null);
    } finally {
      setUploading(false);
      setJobStatus(null);
      setActiveJobId(null);
    }
  }
  async function openHistory(runId) {
    try {
      const data = await apiFetch(
        `/research/${encodeURIComponent(runId)}`,
        {},
        30000,
      );
      setResult(data);
      setQuestion(data.question || "");
      setTitle(data.report_title || "");
      setReportType(data.report_type || "standard_report");
      setChat(hydrateChat(data));
      setStage("report");
      setError("");
      setVersionsOpen(false);
      setMobileChatOpen(true);
      void loadVersions(runId);
    } catch (e) {
      setError(e.message || "Could not open this research case.");
    }
  }
  async function loadVersions(runId = result?.run_id) {
    if (!runId) return;
    try {
      setVersions(
        await apiFetch(
          `/research/${encodeURIComponent(runId)}/versions`,
          {},
          20000,
        ),
      );
    } catch {
      setVersions([]);
    }
  }
  async function updateCase(runId, patch) {
    try {
      await apiFetch(`/research/${encodeURIComponent(runId)}/case`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      await refreshHistory();
    } catch (e) {
      setError(e.message || "Could not update the research case.");
    }
  }
  async function deleteCase(runId) {
    if (
      !window.confirm(
        "Delete this research case and all of its saved versions?",
      )
    )
      return;
    try {
      await apiFetch(`/research/${encodeURIComponent(runId)}/case`, {
        method: "DELETE",
      });
      if (result?.run_id === runId) reset();
      await refreshHistory();
    } catch (e) {
      setError(e.message || "Could not delete the research case.");
    }
  }
  function beginRename(item) {
    setRenamingRunId(item.run_id);
    setRenameText(item.case_name || item.report_title || item.question || "");
    setMenuRunId(null);
  }
  async function commitRename(runId) {
    if (!renameText.trim()) return;
    await updateCase(runId, { name: renameText.trim() });
    setRenamingRunId(null);
    setRenameText("");
  }
  async function download(kind) {
    if (!result?.run_id || downloadingKind) return;
    let objectUrl = "";
    try {
      setDownloadingKind(kind);
      setError("");
      const response = await fetch(
        `${API}/research/${encodeURIComponent(result.run_id)}/download/${kind}`,
        {
          method: "GET",
          cache: "no-store",
          headers: { Accept: kind === "pdf" ? "application/pdf" : "*/*" },
        },
      );
      if (!response.ok) {
        const raw = await response.text();
        let message = "";
        try {
          message = JSON.parse(raw)?.detail || "";
        } catch {}
        throw new Error(message || `Download failed (${response.status})`);
      }
      const blob = await response.blob();
      if (!blob.size) throw new Error("The server returned an empty file.");
      const disposition = response.headers.get("content-disposition") || "";
      const match = disposition.match(
        /filename\\*=(?:UTF-8'')?([^;]+)|filename="?([^";]+)"?/i,
      );
      const encodedName = match?.[1] ? decodeURIComponent(match[1]) : "";
      const filename = (encodedName || match?.[2] || "").replace(
        /^["']|["']$/g,
        "",
      );
      const fallback = `${
        (result.report_title || "research-report")
          .replace(/[^a-z0-9]+/gi, "-")
          .replace(/^-|-$/g, "")
          .toLowerCase() || "research-report"
      }.${kind === "bib" ? "bib" : kind === "latex" ? "tex" : kind}`;
      objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = filename || fallback;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      setError(e.message || "Could not download the report.");
    } finally {
      if (objectUrl)
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
      setDownloadingKind("");
    }
  }
  const processing = (loading || uploading) && jobStatus;
  return (
    <div
      className={`app-shell ${leftOpen ? "sidebar-open" : "sidebar-collapsed"}`}
      style={{ "--chat-width": `${chatWidth}px` }}
    >
      <Sidebar
        open={leftOpen}
        setOpen={setLeftOpen}
        history={history}
        result={result}
        menuRunId={menuRunId}
        setMenuRunId={setMenuRunId}
        renamingRunId={renamingRunId}
        renameText={renameText}
        setRenameText={setRenameText}
        beginRename={beginRename}
        commitRename={commitRename}
        deleteCase={deleteCase}
        updateCase={updateCase}
        openHistory={openHistory}
        newResearch={reset}
      />
      <main className='main-content'>
        <header className='mobile-topbar'>
          <button
            className='icon-btn'
            onClick={() => setLeftOpen(true)}
            aria-label='Open sidebar'
          >
            <Menu size={18} />
          </button>
          <span>ResearchOrbit</span>
          <button
            className='icon-btn mobile-chat-button'
            onClick={() => setMobileChatOpen((v) => !v)}
            aria-label='Toggle chat'
          >
            <MessageCircle size={18} />
          </button>
        </header>
        {error && (
          <div className='error-banner' role='alert'>
            <span>{error}</span>
            <button onClick={() => setError("")} aria-label='Dismiss error'>
              <X size={15} />
            </button>
          </div>
        )}
        <section className='workspace-scroll'>
          {stage === "setup" && (
            <Setup
              question={question}
              setQuestion={setQuestion}
              title={title}
              setTitle={setTitle}
              reportType={reportType}
              setReportType={setReportType}
              start={discoverCandidates}
              loading={loading}
            />
          )}
          {stage === "screening" && (
            <Screening
              candidates={candidates}
              loading={loading}
              jobStatus={jobStatus}
              generate={generateReport}
              back={reset}
              onToggle={toggleCandidate}
              uploadedCandidates={uploadedCandidates}
              onUpload={() => uploadRef.current?.click()}
              uploadBusy={uploading}
            />
          )}
          {stage === "report" && result && (
            <Report
              result={result}
              reportRef={reportRef}
              onSelection={setSelectedText}
              selectedText={selectedText}
              explain={explainSelection}
              reanalyze={(e) => {
                if ([...(e.target.files || [])].length > 2) {
                  e.target.value = "";
                  setError(
                    "You can add at most 2 papers after the report is generated.",
                  );
                  return;
                }
                handleUpload(e);
              }}
              uploading={uploading}
              download={download}
              downloadingKind={downloadingKind}
              versions={versions}
              versionsOpen={versionsOpen}
              setVersionsOpen={setVersionsOpen}
              loadVersion={openHistory}
            />
          )}
        </section>
        <input
          ref={uploadRef}
          className='upload-input-hidden'
          type='file'
          accept='.pdf,application/pdf'
          multiple
          onChange={handleUpload}
        />
      </main>
      <ChatPanel
        result={result}
        chat={chat}
        chatText={chatText}
        setChatText={setChatText}
        chatLoading={chatLoading}
        ask={ask}
        chatMode={chatMode}
        setChatMode={setChatMode}
        selectedText={selectedText}
        explainSelection={explainSelection}
        stopChat={stopChat}
        mobileOpen={mobileChatOpen}
        setMobileOpen={setMobileChatOpen}
        onResize={() => {
          resizingChat.current = true;
          document.body.classList.add("resizing-chat");
        }}
      />
      {processing && (
        <ProcessingOverlay
          status={jobStatus}
          clock={progressClock}
          onCancel={activeJobId ? cancelActiveJob : null}
        />
      )}
    </div>
  );
}
export default App;
