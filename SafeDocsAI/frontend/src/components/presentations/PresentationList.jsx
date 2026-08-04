import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
    AlertTriangle,
    Download,
    FileText,
    Loader2,
    Presentation,
    RefreshCw,
    Trash2,
} from 'lucide-react';

import { Button } from '../ui/Button';
import { cn } from '../../lib/utils';
import { useModalDialog } from '../../hooks/useModalDialog';
import { useLocale } from '../../i18n';
import { resolveApiErrorMessage } from '../../lib/apiError';
import { formatLocaleDate } from '../../lib/locale';
import {
    resolvePresentationErrorMessage,
    resolvePresentationStatus,
    resolveProgress,
    resolveQueuePosition,
    resolveTemplateName,
} from '../../lib/presentations';
import { formatSize } from '../../lib/sources';
import { presentationsService } from '../../services/presentationsService';

const STATUS_BADGE_CLASSES = {
    queued: 'bg-slate-100 text-slate-600',
    generating: 'bg-amber-100 text-amber-700',
    ready: 'bg-green-100 text-green-700',
    error: 'bg-red-100 text-red-700',
};

// Имя файла из Content-Disposition, если сервер его прислал. Своё имя клиент
// придумывает только когда заголовка нет: расширение и транслитерация — дело
// сервера, а не браузера.
//
// .pdf — потому что колоды печатает headless Chrome. Расширение здесь ЗАПАСНОЕ
// и не сверяется с содержимым: колоды, собранные прежним рендерером, приходят
// в .pptx, но приходят они со своим Content-Disposition, до этой строки дело не
// доходит. Сюда попадают только ответы без заголовка вовсе — то есть случай,
// которого в норме не бывает.
const FALLBACK_FILENAME_SUFFIX = '.pdf';

const resolveDownloadFilename = (response, presentationId) => {
    const disposition = String(response?.headers?.['content-disposition'] || '');
    const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match) {
        try {
            return decodeURIComponent(utf8Match[1].trim());
        } catch {
            // Битая кодировка в заголовке — не повод отменять скачивание.
        }
    }

    const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
    if (plainMatch) return plainMatch[1].trim();

    return `presentation-${presentationId}${FALLBACK_FILENAME_SUFFIX}`;
};

/**
 * Список заказанных презентаций.
 *
 * Данные и опрос живут выше (NotebookPresentationsPage): здесь только показ,
 * скачивание и удаление. Ошибки этих двух действий держатся отдельно от ошибки
 * загрузки списка и друг от друга — неудачное скачивание не должно прятать
 * список, а неудачное удаление обязано остаться в диалоге, где его ждут.
 */
const PresentationList = ({
    items,
    total,
    templates,
    isLoading,
    error,
    onRetry,
    onDeleted,
}) => {
    const { locale, t } = useLocale();

    const [deleteTarget, setDeleteTarget] = useState(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [deleteError, setDeleteError] = useState('');
    const [downloadingId, setDownloadingId] = useState(null);
    const [downloadError, setDownloadError] = useState('');

    const deleteDialogRef = useRef(null);
    const deleteCancelRef = useRef(null);

    const templatesByKey = useMemo(
        () => Object.fromEntries((templates || []).map((template) => [template.key, template])),
        [templates],
    );

    const closeDeleteDialog = useCallback(() => {
        // Пока запрос в полёте, закрывать нечего: ответ всё равно придёт, а
        // сообщение об ошибке пользователь бы уже не увидел.
        if (isDeleting) return;
        setDeleteTarget(null);
        setDeleteError('');
    }, [isDeleting]);

    useModalDialog(Boolean(deleteTarget), closeDeleteDialog, deleteDialogRef, deleteCancelRef);

    const handleDownload = async (presentation) => {
        if (downloadingId != null) return;

        try {
            setDownloadingId(presentation.id);
            setDownloadError('');
            const response = await presentationsService.downloadBlob(presentation.id);

            // Скачивание через blob, а не ссылкой: ручка закрыта сессией, и
            // прямой переход по адресу ушёл бы мимо axios-клиента — без
            // обновления протухшего токена и без единой обработки ошибок.
            const objectUrl = URL.createObjectURL(response.data);
            const link = document.createElement('a');
            link.href = objectUrl;
            link.download = resolveDownloadFilename(response, presentation.id);
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(objectUrl);
        } catch (downloadFailure) {
            console.error('Failed to download presentation:', downloadFailure);
            setDownloadError(resolveApiErrorMessage(downloadFailure, t, 'presentations.downloadFailed'));
        } finally {
            setDownloadingId(null);
        }
    };

    const handleDelete = async () => {
        if (!deleteTarget || isDeleting) return;

        try {
            setIsDeleting(true);
            setDeleteError('');
            await presentationsService.delete(deleteTarget.id);
            onDeleted?.(deleteTarget.id);
            setDeleteTarget(null);
        } catch (deleteFailure) {
            console.error('Failed to delete presentation:', deleteFailure);
            setDeleteError(resolveApiErrorMessage(deleteFailure, t, 'presentations.deleteFailed'));
        } finally {
            setIsDeleting(false);
        }
    };

    const renderStatus = (presentation) => {
        const status = resolvePresentationStatus(presentation.status);

        if (status === 'queued') {
            const position = resolveQueuePosition(presentation);
            return position != null
                ? t('presentations.status.queuedPosition', { position })
                : t('presentations.status.queued');
        }

        if (status === 'generating') {
            return t('presentations.status.generating', { progress: resolveProgress(presentation.progress) });
        }

        if (status === 'ready') return t('presentations.status.ready');

        return t('presentations.status.error', { reason: resolvePresentationErrorMessage(presentation, t) });
    };

    return (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-lg font-semibold text-slate-900">{t('presentations.listTitle')}</h2>
                {items.length > 0 ? (
                    <span className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
                        {t('presentations.listCount', { count: items.length, total: total ?? items.length })}
                    </span>
                ) : null}
            </div>

            {/* Сбой ЗАГРУЗКИ списка показываем баннером над тем, что уже
                получено: обнулять список из-за неудачного обновления — значит
                прятать данные, которые никуда не делись. */}
            {error ? (
                <div role="alert" className="mb-4 flex flex-col gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                    <span className="flex items-center gap-2 font-semibold">
                        <AlertTriangle className="h-4 w-4" />
                        {error}
                    </span>
                    {onRetry ? (
                        <Button type="button" variant="outline" size="sm" className="self-start" onClick={onRetry}>
                            <RefreshCw className="h-4 w-4" />
                            {t('presentations.retry')}
                        </Button>
                    ) : null}
                </div>
            ) : null}

            {downloadError ? (
                <p role="alert" className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                    {downloadError}
                </p>
            ) : null}

            {isLoading && items.length === 0 ? (
                <div role="status" className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {t('presentations.listLoading')}
                </div>
            ) : items.length === 0 ? (
                error ? null : (
                    <div className="flex flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 px-6 py-10 text-center">
                        <div className="rounded-2xl bg-white p-4 text-[#1f3a60] shadow-sm">
                            <Presentation className="h-6 w-6" />
                        </div>
                        <p className="mt-4 text-sm font-semibold text-slate-800">{t('presentations.emptyTitle')}</p>
                        <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{t('presentations.emptyDescription')}</p>
                    </div>
                )
            ) : (
                <ul className="space-y-3">
                    {items.map((presentation) => {
                        const status = resolvePresentationStatus(presentation.status);
                        const progress = resolveProgress(presentation.progress);
                        const templateName = resolveTemplateName(templatesByKey[presentation.template_key], presentation.language)
                            || presentation.template_key;
                        const errorMessage = resolvePresentationErrorMessage(presentation, t);
                        const isGenerating = status === 'generating';

                        return (
                            <li
                                key={presentation.id}
                                className="rounded-2xl border border-slate-200 p-4 transition hover:border-slate-300"
                            >
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div className="min-w-0">
                                        <p className="text-sm font-semibold text-slate-900">{templateName}</p>
                                        <p className="mt-1 text-xs text-slate-400">
                                            {t('presentations.cardMeta', {
                                                id: presentation.id,
                                                slides: presentation.slide_count,
                                                language: String(presentation.language || '').toUpperCase(),
                                            })}
                                        </p>
                                    </div>

                                    {/* Свой role="status" на карточку, а не одна живая
                                        область на весь список: объявлять надо изменение
                                        конкретной презентации, иначе при каждом тике
                                        опроса зачитывался бы список целиком. */}
                                    <span
                                        role="status"
                                        className={cn(
                                            'shrink-0 rounded-full px-3 py-1 text-[11px] font-semibold',
                                            STATUS_BADGE_CLASSES[status],
                                        )}
                                    >
                                        {renderStatus(presentation)}
                                    </span>
                                </div>

                                {isGenerating ? (
                                    <div
                                        className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-100"
                                        role="progressbar"
                                        aria-valuenow={progress}
                                        aria-valuemin={0}
                                        aria-valuemax={100}
                                        aria-label={t('presentations.progressLabel')}
                                    >
                                        <div className="h-full rounded-full bg-[#1f3a60] transition-[width]" style={{ width: `${progress}%` }} />
                                    </div>
                                ) : null}

                                {presentation.description ? (
                                    <p className="mt-3 line-clamp-3 whitespace-pre-wrap break-words text-sm leading-6 text-slate-500">
                                        {presentation.description}
                                    </p>
                                ) : null}

                                {errorMessage ? (
                                    /* error_text остаётся и подсказкой при наведении:
                                       в тексте он появляется только когда код неизвестен
                                       (см. resolvePresentationErrorMessage). */
                                    <p
                                        className="mt-3 flex items-start gap-1.5 rounded-xl bg-red-50 px-3 py-2 text-xs leading-5 text-red-600"
                                        title={presentation.error_text || undefined}
                                    >
                                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                                        <span className="min-w-0 break-words">{errorMessage}</span>
                                    </p>
                                ) : null}

                                <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                                        <span className="inline-flex items-center gap-1.5">
                                            <FileText className="h-3.5 w-3.5" />
                                            {formatSize(presentation.file_size, t)}
                                        </span>
                                        <span>
                                            {t('presentations.createdAt', {
                                                date: formatLocaleDate(presentation.created_at, locale, {
                                                    day: 'numeric',
                                                    month: 'short',
                                                    year: 'numeric',
                                                    hour: '2-digit',
                                                    minute: '2-digit',
                                                }, '—'),
                                            })}
                                        </span>
                                    </div>

                                    <div className="flex flex-wrap items-center gap-2">
                                        {/* Скачивание только у готовой: у остальных файла
                                            на диске ещё нет, и кнопка вела бы в 409/404. */}
                                        {status === 'ready' ? (
                                            <Button
                                                type="button"
                                                variant="outline"
                                                size="sm"
                                                onClick={() => handleDownload(presentation)}
                                                isLoading={downloadingId === presentation.id}
                                            >
                                                <Download className="h-4 w-4" />
                                                {t('presentations.download')}
                                            </Button>
                                        ) : null}

                                        {/* Удаление во время генерации сервер отвергнет
                                            конфликтом (presentation.generation_in_progress).
                                            Кнопка выключена заранее и объясняет почему:
                                            403/409 в ответ на разрешённое действие — это
                                            сюрприз, а не сообщение. */}
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="sm"
                                            className="text-red-600 hover:bg-red-50 hover:text-red-700"
                                            disabled={isGenerating}
                                            title={isGenerating
                                                ? t('presentations.deleteDisabledGenerating')
                                                : t('presentations.delete')}
                                            aria-label={isGenerating
                                                ? t('presentations.deleteDisabledGenerating')
                                                : t('presentations.delete')}
                                            onClick={() => {
                                                setDeleteError('');
                                                setDeleteTarget(presentation);
                                            }}
                                        >
                                            <Trash2 className="h-4 w-4" />
                                            {t('presentations.delete')}
                                        </Button>
                                    </div>
                                </div>

                                {isGenerating ? (
                                    <p className="mt-2 text-xs text-slate-400">{t('presentations.deleteDisabledGenerating')}</p>
                                ) : null}
                            </li>
                        );
                    })}
                </ul>
            )}

            {deleteTarget ? (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div className="absolute inset-0 bg-black/50" onClick={closeDeleteDialog} aria-hidden="true" />

                    <div
                        ref={deleteDialogRef}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="presentation-delete-title"
                        aria-describedby="presentation-delete-description"
                        className="relative w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-xl"
                    >
                        <div className="flex items-start gap-3 px-6 py-5">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-red-100 text-red-600">
                                <AlertTriangle className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                                <h3 id="presentation-delete-title" className="text-base font-semibold text-slate-900">
                                    {t('presentations.deleteConfirmTitle')}
                                </h3>
                                <p id="presentation-delete-description" className="mt-1 break-words text-sm text-slate-500">
                                    {t('presentations.deleteConfirmDescription', {
                                        name: resolveTemplateName(templatesByKey[deleteTarget.template_key], deleteTarget.language)
                                            || deleteTarget.template_key,
                                        id: deleteTarget.id,
                                    })}
                                </p>
                            </div>
                        </div>

                        {deleteError ? (
                            <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                                {deleteError}
                            </p>
                        ) : null}

                        <div className="flex justify-end gap-2 border-t border-slate-200 px-6 py-4">
                            <Button
                                ref={deleteCancelRef}
                                type="button"
                                variant="outline"
                                onClick={closeDeleteDialog}
                                disabled={isDeleting}
                            >
                                {t('presentations.cancel')}
                            </Button>
                            <Button type="button" variant="destructive" onClick={handleDelete} isLoading={isDeleting}>
                                {isDeleting ? t('presentations.deleting') : t('presentations.delete')}
                            </Button>
                        </div>
                    </div>
                </div>
            ) : null}
        </section>
    );
};

export default PresentationList;
