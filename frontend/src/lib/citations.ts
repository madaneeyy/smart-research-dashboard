import type { ResearchItem, WorkspaceSource } from "./api";

export interface ArxivCitation {
  title: string;
  authors: string[];
  year: string;
  arxivId: string;
  url: string;
}

function cleanAuthors(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((author) => String(author).trim())
    .filter(Boolean);
}

function extractArxivId(value: string): string {
  const match = value.match(/(?:arxiv\.org\/(?:abs|pdf)\/)?([^?#/]+?)(?:\.pdf)?$/i);
  return match?.[1]?.trim() || value.trim();
}

function getYear(value: string | null | undefined): string {
  if (!value) return "n.d.";
  const match = value.match(/^(\d{4})/);
  return match?.[1] || "n.d.";
}

export function citationFromResearchItem(result: ResearchItem): ArxivCitation {
  const metadata = result.metadata ?? {};
  const rawId =
    typeof metadata.arxiv_id === "string"
      ? metadata.arxiv_id
      : result.id;

  return {
    title: result.title.trim(),
    authors: cleanAuthors(result.authors),
    year: getYear(result.published),
    arxivId: extractArxivId(rawId),
    url: result.url.trim(),
  };
}

export function citationFromWorkspaceSource(
  source: WorkspaceSource,
): ArxivCitation | null {
  if (source.source_type.toLowerCase() !== "arxiv") return null;

  const metadata = source.metadata ?? {};
  const authors = cleanAuthors(metadata.authors);
  const published =
    typeof metadata.published === "string"
      ? metadata.published
      : null;
  const rawId =
    typeof metadata.arxiv_id === "string"
      ? metadata.arxiv_id
      : source.url || source.title;

  return {
    title: source.title.trim(),
    authors,
    year: getYear(published),
    arxivId: extractArxivId(rawId),
    url:
      (typeof metadata.canonical_url === "string"
        ? metadata.canonical_url
        : source.url || "").trim(),
  };
}

function formatApaAuthor(author: string): string {
  const parts = author.trim().split(/\s+/).filter(Boolean);
  if (parts.length <= 1) return author.trim();

  const last = parts.pop() as string;
  const initials = parts
    .map((part) => `${part[0]}.`)
    .join(" ");

  return `${last}, ${initials}`;
}

function escapeBibtex(value: string): string {
  return value
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([&%$#_{}])/g, "\\$1");
}

function citationKey(citation: ArxivCitation): string {
  const firstAuthor = citation.authors[0]
    ?.split(/\s+/)
    .filter(Boolean)
    .pop() || "paper";
  const surname = firstAuthor.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
  const titleWord =
    citation.title
      .replace(/[^a-zA-Z0-9 ]/g, " ")
      .split(/\s+/)
      .find(Boolean)
      ?.toLowerCase() || "paper";

  return `${surname || "paper"}_${titleWord}_${citation.year}`;
}

export function formatApaCitation(citation: ArxivCitation): string {
  const authors = citation.authors.length
    ? citation.authors.map(formatApaAuthor).join(", ")
    : "Unknown author";

  return `${authors} (${citation.year}). ${citation.title}. arXiv. ${citation.url}`;
}

export function formatBibtexCitation(citation: ArxivCitation): string {
  const authors = citation.authors.length
    ? citation.authors.join(" and ")
    : "Unknown author";

  return `@article{${citationKey(citation)},\n  title = {${escapeBibtex(citation.title)}},\n  author = {${escapeBibtex(authors)}},\n  year = {${citation.year}},\n  journal = {arXiv},\n  eprint = {${escapeBibtex(citation.arxivId)}},\n  url = {${citation.url}}\n}`;
}

export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    if (!document.execCommand("copy")) {
      throw new Error("Copy command failed.");
    }
  } finally {
    document.body.removeChild(textarea);
  }
}
