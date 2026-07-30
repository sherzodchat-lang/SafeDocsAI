import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { X, ChevronLeft, ChevronRight, Loader2, ZoomIn, ZoomOut } from 'lucide-react';
import { sourcesService } from '../services/sourcesService';

pdfjs.GlobalWorkerOptions.workerSrc = 'https://unpkg.com/pdfjs-dist@5.4.296/build/pdf.worker.min.mjs';

const DocumentViewer = ({ docId, docName, chunkId, page, onClose }) => {
  const [fileBlob, setFileBlob] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [textContent, setTextContent] = useState('');
  const [chunkContext, setChunkContext] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // PDF state
  const [numPages, setNumPages] = useState(null);
  const [currentPage, setCurrentPage] = useState(page || 1);
  const [scale, setScale] = useState(1.2);

  const highlightRef = useRef(null);

  const ext = useMemo(() => {
    if (!docName) return '';
    return (docName.split('.').pop() || '').toLowerCase();
  }, [docName]);

  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError('');
      try {
        const [blobRes, contextRes] = await Promise.allSettled([
          sourcesService.getPreviewBlob(docId),
          chunkId ? sourcesService.getChunkContext(docId, chunkId) : Promise.resolve(null),
        ]);

        if (cancelled) return;

        if (blobRes.status === 'fulfilled') {
          const blob = blobRes.value.data;
          const contentType = blobRes.value.headers?.['content-type'] || '';
          if (contentType.includes('pdf') || ext === 'pdf') {
            setFileType('pdf');
            setFileBlob(blob);
            setFileUrl(URL.createObjectURL(blob));
          } else {
            setFileType('text');
            const text = await blob.text();
            setTextContent(text);
          }
        } else {
          const reason = blobRes.reason;
          const status = reason?.response?.status;
          let detail = '';
          try {
            const errData = reason?.response?.data;
            if (errData instanceof Blob) {
              detail = JSON.parse(await errData.text())?.detail || '';
            } else {
              detail = errData?.detail || '';
            }
          } catch { /* ignore */ }
          setError(detail || `Не удалось загрузить файл (${status || 'ошибка сети'})`);
        }

        if (contextRes.status === 'fulfilled' && contextRes.value?.data) {
          setChunkContext(contextRes.value.data);
        }
      } catch {
        if (!cancelled) setError('Ошибка загрузки');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; setFileUrl((u) => { if (u) URL.revokeObjectURL(u); return null; }); };
  }, [docId, chunkId, ext]);

  useEffect(() => {
    if (page) setCurrentPage(page);
  }, [page]);

  // Scroll to highlight for text viewer
  useEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [chunkContext, textContent]);

  const onDocumentLoadSuccess = useCallback(({ numPages: n }) => {
    setNumPages(n);
    if (page && page <= n) setCurrentPage(page);
  }, [page]);

  const highlightText = useMemo(() => {
    if (!chunkContext) return null;
    const target = chunkContext.find((c) => c.highlight);
    return target?.text || null;
  }, [chunkContext]);

  const renderTextContent = () => {
    if (!highlightText) {
      return <pre className="whitespace-pre-wrap p-6 font-mono text-sm text-slate-700">{textContent}</pre>;
    }

    const idx = textContent.indexOf(highlightText.trim().slice(0, 100));
    if (idx === -1) {
      // Fallback — show chunk context directly
      return (
        <div className="space-y-2 p-6">
          {chunkContext.map((chunk) => (
            <div
              key={chunk.chunk_id}
              ref={chunk.highlight ? highlightRef : null}
              className={`rounded-lg p-4 font-mono text-sm ${
                chunk.highlight
                  ? 'border-2 border-amber-400 bg-amber-50 text-slate-900'
                  : 'bg-slate-50 text-slate-600'
              }`}
            >
              <div className="mb-1 text-xs text-slate-400">
                стр. {chunk.page}{chunk.section ? ` · ${chunk.section}` : ''}
              </div>
              <pre className="whitespace-pre-wrap">{chunk.text}</pre>
            </div>
          ))}
        </div>
      );
    }

    const before = textContent.slice(Math.max(0, idx - 500), idx);
    const match = textContent.slice(idx, idx + highlightText.trim().length);
    const after = textContent.slice(idx + highlightText.trim().length, idx + highlightText.trim().length + 500);

    return (
      <pre className="whitespace-pre-wrap p-6 font-mono text-sm text-slate-700">
        {before.length > 0 && <span>{'...'}{before}</span>}
        <mark ref={highlightRef} className="rounded bg-amber-200 px-0.5">{match}</mark>
        {after.length > 0 && <span>{after}{'...'}</span>}
      </pre>
    );
  };

  const renderPdfHighlight = useCallback(() => {
    if (!highlightText) return null;
    return (
      <div className="mt-4 rounded-lg border-2 border-amber-400 bg-amber-50 p-4">
        <div className="mb-1 text-xs font-semibold text-amber-700">Выделенный фрагмент:</div>
        <p className="text-sm text-slate-800">{highlightText.slice(0, 500)}{highlightText.length > 500 ? '...' : ''}</p>
      </div>
    );
  }, [highlightText]);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4">
      <div className="flex h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold text-slate-800">{docName || 'Документ'}</h3>
            {page && <span className="text-xs text-slate-400">стр. {page}</span>}
          </div>
          <button
            onClick={onClose}
            className="ml-3 rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {loading && (
            <div className="flex h-full items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
            </div>
          )}

          {error && (
            <div className="flex h-full items-center justify-center text-sm text-red-500">{error}</div>
          )}

          {!loading && !error && fileType === 'text' && renderTextContent()}

          {!loading && !error && fileType === 'pdf' && fileUrl && (
            <div className="flex flex-col items-center">
              {renderPdfHighlight()}
              <Document
                file={fileUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={<Loader2 className="mx-auto mt-12 h-8 w-8 animate-spin text-slate-400" />}
              >
                <Page pageNumber={currentPage} scale={scale} />
              </Document>
            </div>
          )}
        </div>

        {/* Footer — PDF controls */}
        {fileType === 'pdf' && numPages && (
          <div className="flex items-center justify-center gap-3 border-t border-slate-200 px-4 py-2">
            <button
              onClick={() => setScale((s) => Math.max(0.5, s - 0.2))}
              className="rounded p-1 text-slate-500 hover:bg-slate-100"
              title="Уменьшить"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              onClick={() => setScale((s) => Math.min(3, s + 0.2))}
              className="rounded p-1 text-slate-500 hover:bg-slate-100"
              title="Увеличить"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <div className="mx-2 h-4 w-px bg-slate-200" />
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="rounded p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-xs text-slate-500">
              {currentPage} / {numPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
              disabled={currentPage >= numPages}
              className="rounded p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;
