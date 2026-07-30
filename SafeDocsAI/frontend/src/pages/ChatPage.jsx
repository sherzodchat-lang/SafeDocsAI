import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Paperclip, Shield, ThumbsDown, ThumbsUp, User } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { chatService } from '../services/chatService';
import { getSessionUsername } from '../services/api';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import DocumentViewer from '../components/DocumentViewer';
import { cn } from '../lib/utils';
import { useLocale } from '../i18n';
import { formatLocaleDate } from '../lib/locale';

const CHAT_HISTORY_STORAGE_PREFIX = 'knowledgeai.chat.history.';
const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';
const MAX_PERSISTED_MESSAGES = 100;
const PENDING_MESSAGE_TTL_MS = 600000; // 10 минут — LLM отвечает до 5+ мин
// Имя пользователя разделяет локальную историю чата. Раньше его доставали из
// JWT в localStorage; теперь токен лежит в httpOnly-куке и из JS не читается,
// поэтому имя запоминается при входе отдельно.
const resolveCurrentUsername = () => getSessionUsername() || 'anonymous';

const getChatStorageKey = (scope) => `${CHAT_HISTORY_STORAGE_PREFIX}${resolveCurrentUsername()}.${scope}`;

const createRequestId = () => {
    const now = Date.now();
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return `req-${now}-${crypto.randomUUID()}`;
    }
    return `req-${now}-${Math.random().toString(16).slice(2)}`;
};

const getCreatedAtFromRequestId = (requestId) => {
    if (typeof requestId !== 'string') return null;
    const match = requestId.match(/^req-(\d+)-/);
    if (!match) return null;
    const parsed = Number(match[1]);
    return Number.isFinite(parsed) ? parsed : null;
};

const normalizeMessage = (message) => {
    if (!message || typeof message !== 'object') return null;
    if (message.role !== 'assistant' && message.role !== 'user') return null;
    if (typeof message.content !== 'string') return null;

    const normalized = {
        role: message.role,
        content: message.content,
    };

    if (message.pending === true) {
        normalized.pending = true;
        normalized.requestId = typeof message.requestId === 'string' ? message.requestId : createRequestId();
        normalized.createdAt = typeof message.createdAt === 'number' ? message.createdAt : Date.now();
        if (message.streaming === true) {
            normalized.streaming = true;
        }
    }

    if (Array.isArray(message.sources)) {
        normalized.sources = message.sources;
    }

    if (typeof message.logId === 'number') {
        normalized.logId = message.logId;
    }

    if (message.feedback === 'up' || message.feedback === 'down') {
        normalized.feedback = message.feedback;
    }

    return normalized;
};

const normalizeMessages = (messages, initialAssistantMessage) => {
    const sanitized = Array.isArray(messages)
        ? messages.map(normalizeMessage).filter(Boolean)
        : [];

    if (sanitized.length === 0) return [initialAssistantMessage];
    if (sanitized[0].role !== 'assistant') return [initialAssistantMessage, ...sanitized];
    return sanitized;
};

const replacePendingMessageByRequestId = (messages, requestId, nextMessage) => {
    if (!requestId) return [...messages, nextMessage];

    let replaced = false;
    const updated = messages.map((message) => {
        if (!replaced && message.pending === true && message.requestId === requestId) {
            replaced = true;
            return nextMessage;
        }
        return message;
    });

    if (!replaced) {
        updated.push(nextMessage);
    }

    return updated;
};

const updatePendingMessageByRequestId = (messages, requestId, updater) => {
    if (!requestId) return messages;

    return messages.map((message) => {
        if (message.pending === true && message.requestId === requestId) {
            return updater(message);
        }
        return message;
    });
};

const removePendingMessageByRequestId = (messages, requestId) => {
    if (!requestId) return messages;

    let removed = false;
    return messages.filter((message) => {
        if (!removed && message.pending === true && message.requestId === requestId) {
            removed = true;
            return false;
        }
        return true;
    });
};

const resolvePendingMessages = (messages, interruptedText) => {
    const now = Date.now();
    let changed = false;

    const updated = messages.map((message) => {
        if (message.pending !== true) return message;
        const createdAt = typeof message.createdAt === 'number' ? message.createdAt : now;
        if (now - createdAt <= PENDING_MESSAGE_TTL_MS) return message;
        changed = true;
        return {
            role: 'assistant',
            content: interruptedText,
        };
    });

    return { messages: updated, changed };
};

const loadMessagesFromStorage = (storageKey, initialAssistantMessage, interruptedText) => {
    if (typeof window === 'undefined') return [initialAssistantMessage];

    try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) return [initialAssistantMessage];
        const normalized = normalizeMessages(JSON.parse(raw), initialAssistantMessage);
        const { messages: resolved, changed } = resolvePendingMessages(normalized, interruptedText);
        if (changed) {
            localStorage.setItem(storageKey, JSON.stringify(resolved));
        }
        return resolved;
    } catch {
        return [initialAssistantMessage];
    }
};

const persistMessagesToStorage = (storageKey, messages, initialAssistantMessage) => {
    if (typeof window === 'undefined') return;

    try {
        const normalized = normalizeMessages(messages, initialAssistantMessage).slice(-MAX_PERSISTED_MESSAGES);
        localStorage.setItem(storageKey, JSON.stringify(normalized));
    } catch (error) {
        console.error('Failed to persist chat history', error);
    }
};

const renderInlineMarkdown = (text) => {
    const parts = (text || '').split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g).filter(Boolean);

    return parts.map((part, index) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
        }

        if (part.startsWith('`') && part.endsWith('`')) {
            return (
                <code key={`${part}-${index}`} className="rounded bg-slate-100 px-1 py-0.5 text-slate-700">
                    {part.slice(1, -1)}
                </code>
            );
        }

        const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (linkMatch) {
            const [, label, href] = linkMatch;
            return (
                <a
                    key={`${part}-${index}`}
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="font-semibold text-[#1f3a60] hover:underline"
                >
                    {label}
                </a>
            );
        }

        return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    });
};

const MarkdownContent = ({ content }) => {
    const lines = (content || '').split('\n');

    return (
        <div className="min-w-0 space-y-1.5 overflow-hidden break-words leading-relaxed">
            {lines.map((line, index) => {
                if (line.startsWith('### ')) {
                    return <h3 key={index} className="text-sm font-semibold">{renderInlineMarkdown(line.slice(4))}</h3>;
                }
                if (line.startsWith('## ')) {
                    return <h2 key={index} className="text-base font-semibold">{renderInlineMarkdown(line.slice(3))}</h2>;
                }
                if (line.startsWith('# ')) {
                    return <h1 key={index} className="text-lg font-bold">{renderInlineMarkdown(line.slice(2))}</h1>;
                }
                if (line.startsWith('- ')) {
                    return (
                        <div key={index} className="flex items-start gap-2 pl-1">
                            <span className="mt-1">•</span>
                            <span>{renderInlineMarkdown(line.slice(2))}</span>
                        </div>
                    );
                }
                return <p key={index} className="whitespace-pre-wrap">{renderInlineMarkdown(line)}</p>;
            })}
        </div>
    );
};

const formatTodayLabel = (locale, t) => {
    const date = new Date();
    const formatted = formatLocaleDate(date, locale, {
        month: 'long',
        day: 'numeric',
    });
    return t('chat.today', { date: formatted });
};

const resolveNotebookId = (notebookId) => {
    if (notebookId !== undefined) {
        return notebookId == null ? null : Number(notebookId);
    }

    const storedValue = localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
    return storedValue ? Number(storedValue) : null;
};

const formatNotebookLabel = (notebookId, t) => {
    if (notebookId == null) return t('chat.allSources');
    return String(notebookId);
};

const ChatPage = ({ notebookId, mode = 'page' }) => {
    const { locale, t } = useLocale();
    const { register, handleSubmit: formHandleSubmit, reset } = useForm();
    const effectiveNotebookId = resolveNotebookId(notebookId);
    const chatScope = effectiveNotebookId == null ? 'global' : `notebook.${effectiveNotebookId}`;
    const chatStorageKey = getChatStorageKey(chatScope);
    const initialAssistantMessage = useMemo(() => ({
        role: 'assistant',
        content: t('chat.welcome'),
    }), [t]);
    const quickQuestions = useMemo(() => ([
        t('chat.quickQuestions.summary'),
        t('chat.quickQuestions.topics'),
        t('chat.quickQuestions.definitions'),
        t('chat.quickQuestions.bestSources'),
    ]), [t]);
    const [messages, setMessages] = useState(() => loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted')));
    const [viewerSource, setViewerSource] = useState(null);
    const messagesEndRef = useRef(null);
    const isPageLeavingRef = useRef(false);
    const isNotebookPanel = mode === 'notebookPanel';

    const hasPendingMessage = messages.some((message) => message.pending === true);
    const hasConversation = messages.some((message, index) => index > 0 && !message.pending);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        setMessages(loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted')));
    }, [chatStorageKey, initialAssistantMessage, t]);

    useEffect(() => {
        persistMessagesToStorage(chatStorageKey, messages, initialAssistantMessage);
    }, [chatStorageKey, initialAssistantMessage, messages]);

    useEffect(() => {
        if (notebookId !== undefined) return;

        const handleStorage = (event) => {
            if (event.key === ACTIVE_NOTEBOOK_STORAGE_KEY) {
                const nextNotebookId = resolveNotebookId(undefined);
                const nextScope = nextNotebookId == null ? 'global' : `notebook.${nextNotebookId}`;
                setMessages(loadMessagesFromStorage(getChatStorageKey(nextScope), initialAssistantMessage, t('chat.interrupted')));
            }
        };

        window.addEventListener('storage', handleStorage);
        return () => window.removeEventListener('storage', handleStorage);
    }, [initialAssistantMessage, notebookId, t]);

    useEffect(() => {
        isPageLeavingRef.current = false;

        const markPageLeaving = () => {
            isPageLeavingRef.current = true;
        };

        window.addEventListener('beforeunload', markPageLeaving);
        window.addEventListener('pagehide', markPageLeaving);

        return () => {
            markPageLeaving();
            window.removeEventListener('beforeunload', markPageLeaving);
            window.removeEventListener('pagehide', markPageLeaving);
        };
    }, []);

    const submitMessage = async (messageText) => {
        const cleanMessage = (messageText || '').trim();
        if (!cleanMessage || hasPendingMessage) return;

        const requestId = createRequestId();
        const userMessage = { role: 'user', content: cleanMessage };
        const pendingMessage = {
            role: 'assistant',
            content: t('chat.pending'),
            pending: true,
            requestId,
            createdAt: getCreatedAtFromRequestId(requestId),
        };

        setMessages((prev) => {
            const updated = [...prev, userMessage, pendingMessage];
            persistMessagesToStorage(chatStorageKey, updated, initialAssistantMessage);
            return updated;
        });
        reset();

        let streamedAnswer = '';
        try {
            const finalPayload = await chatService.streamMessage(cleanMessage, effectiveNotebookId, {
                onToken: (token) => {
                    if (!token) return;
                    streamedAnswer += token;
                    setMessages((prev) => updatePendingMessageByRequestId(prev, requestId, (message) => ({
                        ...message,
                        content: streamedAnswer,
                        streaming: true,
                    })));
                },
            });
            const botMessage = {
                role: 'assistant',
                content: finalPayload?.answer || streamedAnswer,
                sources: finalPayload?.sources || [],
                logId: finalPayload?.log_id,
            };

            const stored = loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted'));
            const storageUpdated = replacePendingMessageByRequestId(stored, requestId, botMessage);
            persistMessagesToStorage(chatStorageKey, storageUpdated, initialAssistantMessage);
            setMessages((prev) => replacePendingMessageByRequestId(prev, requestId, botMessage));
        } catch (error) {
            console.error(error);
            const isInterruptedRequest =
                isPageLeavingRef.current ||
                document.visibilityState === 'hidden' ||
                error?.code === 'ERR_CANCELED' ||
                error?.name === 'CanceledError';
            if (isInterruptedRequest) {
                const stored = loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted'));
                const storageUpdated = removePendingMessageByRequestId(stored, requestId);
                persistMessagesToStorage(chatStorageKey, storageUpdated, initialAssistantMessage);
                setMessages((prev) => removePendingMessageByRequestId(prev, requestId));
                return;
            }
            // Негодный, отозванный или просроченный токен приходит как 401;
            // 403 теперь означает нехватку прав, а не мёртвую сессию.
            const isAuthError = error.response?.status === 401;
            if (streamedAnswer.trim() && !isAuthError) {
                const partialMessage = {
                    role: 'assistant',
                    content: streamedAnswer,
                    sources: error.response?.data?.sources || [],
                    logId: error.response?.data?.log_id,
                };
                const stored = loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted'));
                const storageUpdated = replacePendingMessageByRequestId(stored, requestId, partialMessage);
                persistMessagesToStorage(chatStorageKey, storageUpdated, initialAssistantMessage);
                setMessages((prev) => replacePendingMessageByRequestId(prev, requestId, partialMessage));
                return;
            }
            const errorContent = isAuthError
                ? t('chat.authExpired')
                : t('chat.requestFailed');
            const errorMessage = { role: 'assistant', content: errorContent };
            const stored = loadMessagesFromStorage(chatStorageKey, initialAssistantMessage, t('chat.interrupted'));
            const storageUpdated = replacePendingMessageByRequestId(stored, requestId, errorMessage);
            persistMessagesToStorage(chatStorageKey, storageUpdated, initialAssistantMessage);
            setMessages((prev) => replacePendingMessageByRequestId(prev, requestId, errorMessage));
        }
    };

    const onSubmit = async (data) => {
        await submitMessage(data.message);
    };

    const handleFormSubmit = (event) => {
        void formHandleSubmit(onSubmit)(event);
    };

    const handleQuickQuestion = async (question) => {
        if (hasPendingMessage) return;
        await submitMessage(question);
    };

    const handleClearChat = () => {
        if (hasPendingMessage) return;
        setMessages([initialAssistantMessage]);
        reset();
    };

    const handleFeedback = async (logId, rating, index) => {
        if (!logId) return;

        try {
            await chatService.sendFeedback(logId, rating);
            setMessages((prev) => {
                const next = [...prev];
                next[index] = { ...next[index], feedback: rating };
                return next;
            });
        } catch (error) {
            console.error('Failed to send feedback', error);
        }
    };

    const renderSources = (sources) => {
        if (!Array.isArray(sources) || sources.length === 0) return null;

        return (
            <details className="mt-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 text-xs text-slate-500" style={{ containIntrinsicSize: 'auto', maxWidth: '100%' }}>
                <summary className="cursor-pointer list-none font-semibold uppercase tracking-[0.08em] text-[#1f3a60]">
                    {t('chat.sources')} ({sources.length})
                </summary>
                <div className="mt-2 space-y-1">
                    {sources.map((source, sourceIdx) => {
                        if (typeof source === 'string') {
                            return <div key={sourceIdx} className="truncate">{t('chat.sourceItem', { value: source })}</div>;
                        }

                        const docName = source.doc_name || t('chat.sourceFallback', { id: source.doc_id ?? 'N/A' });
                        const page = source.page ? t('chat.sourcePage', { page: source.page }) : '';

                        return (
                            <button
                                key={source.chunk_id || sourceIdx}
                                type="button"
                                onClick={() => setViewerSource({
                                    docId: source.doc_id,
                                    docName: source.doc_name,
                                    chunkId: source.chunk_id,
                                    page: source.page,
                                })}
                                className="block w-full cursor-pointer truncate rounded-md px-2 py-1 text-left transition hover:bg-slate-100"
                            >
                                <span className="font-medium text-[#1f3a60]">{docName}</span>
                                <span className="text-slate-400">{page}</span>
                            </button>
                        );
                    })}
                </div>
            </details>
        );
    };

    return (
        <>
        <div className="h-full w-full min-w-0 overflow-hidden">
            <div className={cn('flex h-full flex-col overflow-hidden bg-white', !isNotebookPanel && 'lg:border-l lg:border-slate-200')}>
                <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
                    {isNotebookPanel ? (
                        <>
                            <div>
                                <h3 className="text-lg font-semibold text-slate-900">{t('chat.notebookChatTitle')}</h3>
                                <p className="mt-1 text-sm text-slate-500">{t('chat.notebookChatDescription')}</p>
                            </div>
                            <Button type="button" variant="outline" size="sm" disabled title={t('chat.sessionsUnavailable')}>
                                {t('chat.sessions')}
                            </Button>
                        </>
                    ) : (
                        <>
                            <div className="flex flex-wrap items-center gap-2">
                                <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-500">
                                    {formatTodayLabel(locale, t)}
                                </div>
                                <div className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-semibold text-[#1f3a60]">
                                    {t('chat.notebookLabel', { value: formatNotebookLabel(effectiveNotebookId, t) })}
                                </div>
                            </div>
                            <Button variant="ghost" size="sm" onClick={handleClearChat} disabled={hasPendingMessage}>
                                {t('chat.clear')}
                            </Button>
                        </>
                    )}
                </div>

                <div className={cn('scrollbar-soft flex-1 overflow-y-auto overflow-x-hidden px-4 py-6 sm:px-8', isNotebookPanel ? 'bg-slate-50' : 'space-y-6 bg-[#f6f8fc]')}>
                    {isNotebookPanel && !hasConversation ? (
                        <div className="flex h-full min-h-[260px] flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-white px-6 text-center shadow-sm">
                            <div className="rounded-2xl bg-[#1f3a60]/10 p-4 text-[#1f3a60]">
                                <Shield className="h-6 w-6" />
                            </div>
                            <h4 className="mt-4 text-lg font-semibold text-slate-900">{t('chat.emptyTitle')}</h4>
                            <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                                {t('chat.emptyDescription')}
                            </p>
                        </div>
                    ) : (
                        <div className="w-full min-w-0 space-y-6">
                            {messages.map((msg, idx) => (
                                <div
                                    key={`${msg.role}-${idx}`}
                                    className={cn(
                                        'flex w-full min-w-0 items-start gap-3',
                                        msg.role === 'user' ? 'justify-end' : 'justify-start',
                                    )}
                                >
                                    {msg.role === 'assistant' && (
                                        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#1f3a60] text-[#c5a059]">
                                            <Shield className="h-4 w-4" />
                                        </div>
                                    )}

                                    <div
                                        className={cn(
                                            'min-w-0 max-w-[85%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed shadow-sm sm:max-w-[74%] overflow-hidden',
                                            msg.role === 'user'
                                                ? 'rounded-tr-md bg-[#1f3a60] text-white'
                                                : 'rounded-tl-md border border-slate-200 bg-[#f2f4f7] text-slate-700',
                                            msg.pending === true && msg.streaming !== true && 'animate-pulse',
                                        )}
                                    >
                                        {msg.role === 'assistant'
                                            ? <MarkdownContent content={msg.content} />
                                            : <div className="whitespace-pre-wrap">{msg.content}</div>}

                                        {renderSources(msg.sources)}

                                        {msg.role === 'assistant' && msg.logId && (
                                            <div className="mt-3 flex gap-2 border-t border-slate-200 pt-2">
                                                <button
                                                    onClick={() => handleFeedback(msg.logId, 'up', idx)}
                                                    className={cn(
                                                        'rounded-md p-1.5 transition',
                                                        msg.feedback === 'up' ? 'bg-green-100 text-green-600' : 'text-slate-400 hover:bg-slate-100',
                                                    )}
                                                    title={t('chat.feedbackGood')}
                                                >
                                                    <ThumbsUp className="h-4 w-4" />
                                                </button>
                                                <button
                                                    onClick={() => handleFeedback(msg.logId, 'down', idx)}
                                                    className={cn(
                                                        'rounded-md p-1.5 transition',
                                                        msg.feedback === 'down' ? 'bg-red-100 text-red-600' : 'text-slate-400 hover:bg-slate-100',
                                                    )}
                                                    title={t('chat.feedbackBad')}
                                                >
                                                    <ThumbsDown className="h-4 w-4" />
                                                </button>
                                            </div>
                                        )}
                                    </div>

                                    {msg.role === 'user' && (
                                        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500">
                                            <User className="h-4 w-4" />
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                <div className="border-t border-slate-200 bg-white px-4 py-4 sm:px-6">
                    {isNotebookPanel ? (
                        <>
                            <form onSubmit={handleFormSubmit} className="relative">
                                <Paperclip className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                                <Input
                                    className="h-12 rounded-2xl border-slate-300 bg-slate-50 pl-10 pr-14 focus:bg-white"
                                    placeholder={t('chat.notebookPlaceholder')}
                                    autoComplete="off"
                                    {...register('message')}
                                />
                                <Button
                                    type="submit"
                                    size="icon"
                                    className="absolute right-2 top-1/2 h-9 w-9 -translate-y-1/2 rounded-xl"
                                    disabled={hasPendingMessage}
                                >
                                    <ArrowRight className="h-4 w-4" />
                                </Button>
                            </form>
                        </>
                    ) : (
                        <>
                            <div className="mb-3 flex flex-wrap gap-2">
                                {quickQuestions.map((question) => (
                                    <button
                                        key={question}
                                        type="button"
                                        onClick={() => handleQuickQuestion(question)}
                                        disabled={hasPendingMessage}
                                        className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-100 disabled:opacity-55"
                                    >
                                        {question}
                                    </button>
                                ))}
                            </div>

                            <form onSubmit={handleFormSubmit} className="relative">
                                <Paperclip className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                                <Input
                                    className="h-12 rounded-xl border-slate-300 bg-slate-50 pl-10 pr-14 focus:bg-white"
                                    placeholder={t('chat.pagePlaceholder')}
                                    autoComplete="off"
                                    {...register('message')}
                                />
                                <Button
                                    type="submit"
                                    size="icon"
                                    className="absolute right-2 top-1/2 h-9 w-9 -translate-y-1/2 rounded-lg"
                                    disabled={hasPendingMessage}
                                >
                                    <ArrowRight className="h-4 w-4" />
                                </Button>
                            </form>

                            <p className="mt-2 text-center text-[11px] font-medium text-slate-400">
                                {t('chat.disclaimer')}
                            </p>
                        </>
                    )}
                </div>
            </div>
        </div>
        {viewerSource && (
            <DocumentViewer
                docId={viewerSource.docId}
                docName={viewerSource.docName}
                chunkId={viewerSource.chunkId}
                page={viewerSource.page}
                onClose={() => setViewerSource(null)}
            />
        )}
    </>
    );
};

export default ChatPage;
