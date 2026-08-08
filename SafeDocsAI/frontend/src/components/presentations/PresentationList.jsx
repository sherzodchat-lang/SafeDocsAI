import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
    AlertTriangle,
    Download,
    Loader2,
    Plus,
    Presentation,
    RefreshCw,
    Trash2,
} from 'lucide-react';

import PreviewImage from './PreviewImage';
import { Button } from '../ui/Button';
import { cn } from '../../lib/utils';
import { useModalDialog } from '../../hooks/useModalDialog';
import { useLocale } from '../../i18n';
import { resolveApiErrorMessage } from '../../lib/apiError';
import { formatLocaleDate } from '../../lib/locale';
import {
    hasPresentationsInProgress,
    resolvePresentationErrorMessage,
    resolvePresentationStatus,
    resolvePresentationTitle,
    resolveProgress,
    resolveQueuePosition,
    resolveTemplateName,
} from '../../lib/presentations';
import { presentationsService } from '../../services/presentationsService';

const STATUS_BADGE_CLASSES = {
    queued: 'bg-white/90 text-slate-600 ring-1 ring-slate-200',
    generating: 'bg-amber-500 text-white',
    ready: 'bg-white/90 text-green-700 ring-1 ring-green-200',
    error: 'bg-red-600 text-white',
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
 * Сетка заказанных презентаций.
 *
 * **Карточка — опознавательный знак, а не витрина.** Задача пользователя здесь
 * одна: УЗНАТЬ свою колоду среди других, а рассматривать её он будет, открыв.
 * Отсюда плотность: до шести карточек в ряд на широком экране (ориентир —
 * Gamma), и всё, что не помогает узнаванию, из карточки убрано. Осталось
 * ровно три опознавательных признака, и каждый различает колоды по-своему:
 * картинка первой страницы (видно оформление и титул), заголовок темы (о чём
 * колода) и дата заказа (какая из двух похожих). Размер файла и подробное
 * описание оттуда ушли: ни то, ни другое не помогает выбрать нужную из шести.
 *
 * Уменьшение карточки не должно превращаться в уменьшение того, чем
 * пользуются. Поэтому подпись осталась читаемого кегля и в две строки (одной
 * строки заголовку темы не хватает — обрезка съедала бы как раз различающую
 * часть), а действия стали значками БЕЗ подписей, но в кнопках 40x40: подпись
 * ушла в title и aria-label, а площадь нажатия осталась пальцевой. Кнопка
 * размером с текст в 11 пикселей была бы ровно тем, чего просили не делать.
 *
 * **Превью — первая страница САМОЙ колоды**, а не картинка её шаблона: адрес
 * приходит полем preview_url готовой колоды (бэкенд рисует её из PDF лениво и
 * кэширует). У колоды, которая ещё собирается или упала, такого адреса нет — и
 * там показывается превью ШАБЛОНА, то есть оформление, которое заказывали. Это
 * не подмена одного другим: пока своей страницы не существует, единственное
 * правдивое утверждение о будущей колоде — как она будет оформлена, и карточка
 * говорит именно его, приглушив картинку и накрыв её индикатором работы.
 *
 * Нажатие на карточку открывает колоду, а не выделяет её: у списка нет ни
 * множественного выбора, ни второго действия по умолчанию, поэтому карточка —
 * это большая кнопка «Открыть». Технически она именно кнопка, лежащая под
 * содержимым (absolute inset-0): вкладывать <button> в <button> нельзя, а
 * содержимое пропускает клики сквозь себя (pointer-events-none) — так у
 * открытия остаётся и фокус с клавиатуры, и подпись для чтения с экрана.
 *
 * Данные и опрос живут выше (NotebookPresentationsPage): здесь показ, открытие,
 * скачивание, удаление и повторный заказ неудавшейся колоды (последний — тоже
 * через страницу, onReorder: заказ есть заказ, и точка постановки в очередь
 * обязана остаться одна). Ошибки действий держатся отдельно от ошибки загрузки
 * списка и друг от друга — неудачное скачивание не должно прятать список, а
 * неудачное удаление обязано остаться в диалоге, где его ждут.
 */
const PresentationList = ({
    items,
    templates,
    isLoading,
    error,
    onRetry,
    onDeleted,
    onReorder,
    onCreate,
}) => {
    const { locale, t } = useLocale();

    const [deleteTarget, setDeleteTarget] = useState(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [deleteError, setDeleteError] = useState('');
    const [downloadingId, setDownloadingId] = useState(null);
    const [openingId, setOpeningId] = useState(null);
    const [reorderingId, setReorderingId] = useState(null);
    // Одна строка на обе неудачи с файлом (скачать и открыть): оба действия
    // ходят за одним и тем же blob'ом, и два одновременных сообщения об одном
    // отказе только спорили бы между собой. Текст у каждой попытки свой.
    const [fileActionError, setFileActionError] = useState('');

    const deleteDialogRef = useRef(null);
    const deleteCancelRef = useRef(null);

    const templatesByKey = useMemo(
        () => Object.fromEntries((templates || []).map((template) => [template.key, template])),
        [templates],
    );

    /**
     * Чем подписана колода — здесь и в диалоге удаления.
     *
     * Порядок отката, а не одно значение, потому что заголовок появляется не
     * сразу и не у всех:
     *   1. title из ответа — то, о чём колода. Единственная настоящая подпись;
     *   2. пока заказ в очереди или генерируется — прямая пометка, что название
     *      будет позже. Подставить сюда описание пользователя нельзя: он писал
     *      пожелание, а не заголовок, и карточка соврала бы о содержимом;
     *   3. имя ШАБЛОНА — последнее прибежище для упавших колод и для старых,
     *      чей заголовок не удалось восстановить из файла. Оно не различает
     *      колоды между собой, и ровно поэтому стоит последним, а не первым,
     *      как было раньше.
     */
    const resolveCardName = useCallback((presentation) => {
        const fallback = ['queued', 'generating'].includes(resolvePresentationStatus(presentation?.status))
            ? t('presentations.titlePending')
            : resolveTemplateName(templatesByKey[presentation?.template_key], presentation?.language)
                || presentation?.template_key
                || '';
        return resolvePresentationTitle(presentation, fallback);
    }, [t, templatesByKey]);

    const closeDeleteDialog = useCallback(() => {
        // Пока запрос в полёте, закрывать нечего: ответ всё равно придёт, а
        // сообщение об ошибке пользователь бы уже не увидел.
        if (isDeleting) return;
        setDeleteTarget(null);
        setDeleteError('');
    }, [isDeleting]);

    useModalDialog(Boolean(deleteTarget), closeDeleteDialog, deleteDialogRef, deleteCancelRef);

    // Скачивание через blob, а не ссылкой: ручка закрыта сессией, и прямой
    // переход по адресу ушёл бы мимо axios-клиента — без обновления протухшего
    // токена и без единой обработки ошибок.
    const triggerDownload = (objectUrl, filename) => {
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
    };

    // Оба файловых действия ждут одно другое: за blob'ом ходят они по одной и
    // той же ручке, и две параллельные загрузки одного файла — двойная плата
    // за то же самое.
    const isFileActionBusy = downloadingId != null || openingId != null;

    const handleDownload = async (presentation) => {
        if (isFileActionBusy) return;

        try {
            setDownloadingId(presentation.id);
            setFileActionError('');
            const response = await presentationsService.downloadBlob(presentation.id);

            const objectUrl = URL.createObjectURL(response.data);
            triggerDownload(objectUrl, resolveDownloadFilename(response, presentation.id));
            URL.revokeObjectURL(objectUrl);
        } catch (downloadFailure) {
            console.error('Failed to download presentation:', downloadFailure);
            setFileActionError(resolveApiErrorMessage(downloadFailure, t, 'presentations.downloadFailed'));
        } finally {
            setDownloadingId(null);
        }
    };

    /**
     * «Открыть» — колода во вкладке браузера, без похода в «Загрузки».
     *
     * Готовую презентацию сначала смотрят («то ли вышло?») и только потом
     * скачивают или заказывают заново — поэтому открытие и стало действием
     * самой карточки, а не одной из кнопок в ряду.
     *
     * Вкладка открывается СИНХРОННО, до запроса: право на новое окно браузер
     * выдаёт только внутри обработчика клика, а после await оно уже потрачено —
     * window.open вернул бы null у любого блокировщика всплывающих окон.
     * Пустая вкладка на секунды загрузки — честная цена за то, чтобы открытие
     * работало, а не «иногда работало».
     *
     * Тип файла до запроса неизвестен: API не отдаёт расширение, а на дисках
     * живут и .pdf нового рендерера, и .pptx прежнего. Поэтому решение
     * принимается по Content-Type уже полученного blob'а: PDF уходит во
     * вкладку, всё остальное — обычным скачиванием (показывать .pptx вкладке
     * нечем, а «ничего не произошло» хуже, чем файл в «Загрузках»).
     *
     * Object URL открытой вкладки НЕ освобождается намеренно: revoke сломал бы
     * ещё открытый просмотр (PDF-просмотрщик дочитывает документ лениво).
     * Утечка ограничена числом явных нажатий и живёт до перезагрузки страницы —
     * в отличие от превью шаблонов, где blob'ы плодились бы каждым обновлением
     * списка и потому освобождаются обязательно.
     */
    const handleOpen = async (presentation) => {
        if (isFileActionBusy) return;

        const previewWindow = window.open('', '_blank');

        try {
            setOpeningId(presentation.id);
            setFileActionError('');
            const response = await presentationsService.downloadBlob(presentation.id);
            const objectUrl = URL.createObjectURL(response.data);

            if (previewWindow && String(response.data?.type || '').includes('pdf')) {
                previewWindow.location.href = objectUrl;
            } else {
                previewWindow?.close();
                triggerDownload(objectUrl, resolveDownloadFilename(response, presentation.id));
                URL.revokeObjectURL(objectUrl);
            }
        } catch (openFailure) {
            console.error('Failed to open presentation:', openFailure);
            // Пустую вкладку надо закрыть самим: осиротевший about:blank
            // выглядит как зависший просмотр, а не как отказ.
            previewWindow?.close();
            setFileActionError(resolveApiErrorMessage(openFailure, t, 'presentations.openFailed'));
        } finally {
            setOpeningId(null);
        }
    };

    /**
     * Повторный заказ с параметрами неудавшейся презентации.
     *
     * Все тексты отказов советуют «закажите снова» — но без этой кнопки совет
     * означал вспоминать и выставлять четыре параметра руками. Здесь они
     * берутся из самой карточки; проверять их заново не нужно: сервер всё
     * равно ответит своим кодом, если, например, шаблон сняли релизом.
     *
     * Пока в блокноте есть активный заказ, кнопка выключена с объяснением —
     * тот же довод, что у выключенного удаления: сервер ответил бы 409
     * generation_in_progress, а отказ на разрешённую с виду кнопку — сюрприз.
     */
    const reorderBlocked = reorderingId != null || hasPresentationsInProgress(items);

    const handleReorder = async (presentation) => {
        if (!onReorder || reorderBlocked) return;

        try {
            setReorderingId(presentation.id);
            // Ошибку заказа показывает страница (createError): исход повторного
            // заказа — тот же исход заказа, и второго места для него не нужно.
            await onReorder({
                template_key: presentation.template_key,
                language: presentation.language,
                slide_count: presentation.slide_count,
                description: presentation.description || '',
            });
        } finally {
            setReorderingId(null);
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

        // В бейдже — только слово «Ошибка»: причина целиком лежит красным
        // блоком в карточке, и пилюля с тем же предложением растягивалась на
        // всю ширину, повторяя текст дважды.
        return t('presentations.status.error');
    };

    return (
        <div className="space-y-4">
            {/* Сбой ЗАГРУЗКИ списка показываем баннером над тем, что уже
                получено: обнулять список из-за неудачного обновления — значит
                прятать данные, которые никуда не делись. */}
            {error ? (
                <div role="alert" className="flex flex-col gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
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

            {fileActionError ? (
                <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                    {fileActionError}
                </p>
            ) : null}

            {isLoading && items.length === 0 ? (
                <div role="status" className="flex items-center justify-center gap-2 py-16 text-sm text-slate-500">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {t('presentations.listLoading')}
                </div>
            ) : items.length === 0 ? (
                error ? null : (
                    /* Пусто — это не ошибка и не «список из нуля строк»: у
                       раздела в этот момент ровно одно осмысленное действие, и
                       оно стоит прямо здесь, а не только в шапке страницы. */
                    <div className="flex flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 px-6 py-14 text-center">
                        <div className="rounded-2xl bg-white p-4 text-[#1f3a60] shadow-sm">
                            <Presentation className="h-6 w-6" />
                        </div>
                        <p className="mt-4 text-sm font-semibold text-slate-800">{t('presentations.emptyTitle')}</p>
                        <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{t('presentations.emptyDescription')}</p>
                        {onCreate ? (
                            <Button type="button" size="lg" className="mt-5" onClick={onCreate}>
                                <Plus className="h-4 w-4" />
                                {t('presentations.create')}
                            </Button>
                        ) : null}
                    </div>
                )
            ) : (
                /* Плотность сетки. Ширина карточки нигде не задана числом: колонок
                   тем больше, чем шире экран, и каждая ступень выбрана так, чтобы
                   карточка не проваливалась ниже примерно 160 px — а это ширина, на
                   которой ещё читаются две строки заголовка и помещается ряд из трёх
                   кнопок 40x40. Шесть в ряд на самых широких экранах — ориентир,
                   названный заказчиком (столько же в этом месте у Gamma). */
                <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
                    {items.map((presentation) => {
                        const status = resolvePresentationStatus(presentation.status);
                        const template = templatesByKey[presentation.template_key];
                        const cardName = resolveCardName(presentation);
                        const errorMessage = resolvePresentationErrorMessage(presentation, t);
                        const isGenerating = status === 'generating';
                        const isInProgress = isGenerating || status === 'queued';
                        const isReady = status === 'ready';
                        const isOpening = openingId === presentation.id;
                        const progress = resolveProgress(presentation.progress);
                        // Своя первая страница — только у готовой колоды; у остальных
                        // её ещё не существует, и картинкой служит превью заказанного
                        // ОФОРМЛЕНИЯ (см. docstring компонента).
                        const previewUrl = isReady
                            ? presentation.preview_url
                            : template?.preview_url;

                        return (
                            <li
                                key={presentation.id}
                                className={cn(
                                    'group relative flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition',
                                    isReady && 'hover:-translate-y-0.5 hover:border-[#1f3a60]/40 hover:shadow-md',
                                )}
                            >
                                {/* Кнопка «открыть» лежит ПОД содержимым и занимает всю
                                    карточку. Её нет у незавершённых и упавших колод: файла
                                    на диске ещё (или уже) нет, и нажатие ушло бы в 409/404. */}
                                {isReady ? (
                                    <button
                                        type="button"
                                        onClick={() => handleOpen(presentation)}
                                        disabled={isFileActionBusy}
                                        aria-label={t('presentations.openCard', { name: cardName })}
                                        className="absolute inset-0 z-0 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/50 focus-visible:ring-offset-2"
                                    />
                                ) : null}

                                {/* Содержимое пропускает клики сквозь себя — иначе оно
                                    перехватывало бы нажатие у кнопки под ним. Ряд действий
                                    ниже возвращает себе события явно. */}
                                <div className="pointer-events-none relative">
                                    <PreviewImage
                                        previewUrl={previewUrl}
                                        alt={isReady
                                            ? t('presentations.deckPreviewAlt', { name: cardName })
                                            : t('presentations.previewAlt', {
                                                name: resolveTemplateName(template, presentation.language)
                                                    || presentation.template_key,
                                            })}
                                        className={cn(
                                            'aspect-[16/9] w-full border-b border-slate-100',
                                            !isReady && 'opacity-60',
                                        )}
                                        imageClassName="object-cover object-top"
                                    />

                                    <span
                                        role="status"
                                        className={cn(
                                            'absolute left-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-semibold shadow-sm',
                                            STATUS_BADGE_CLASSES[status],
                                        )}
                                    >
                                        {renderStatus(presentation)}
                                    </span>

                                    {/* Генерация видна на самой карточке: крутящийся
                                        индикатор поверх превью и полоса прогресса по
                                        нижнему краю. Колода делается минуты, и «чем
                                        занято ожидание» должно читаться без всякого
                                        текста. */}
                                    {isInProgress ? (
                                        <span className="absolute inset-0 flex items-center justify-center bg-white/45">
                                            <Loader2 className="h-6 w-6 animate-spin text-[#1f3a60]" />
                                        </span>
                                    ) : null}

                                    {isGenerating ? (
                                        /* Полоса — только картинка к проценту: число уже
                                           объявляет бейдж статуса (role="status"), и второй
                                           говорящий элемент читал бы то же самое дважды. */
                                        <span aria-hidden="true" className="absolute inset-x-0 bottom-0 block h-1.5 bg-slate-200/80">
                                            <span
                                                className="block h-full bg-amber-500 transition-[width] duration-700 ease-out"
                                                style={{ width: `${progress}%` }}
                                            />
                                        </span>
                                    ) : null}

                                    {isOpening ? (
                                        <span className="absolute inset-0 flex items-center justify-center bg-white/70">
                                            <Loader2 className="h-6 w-6 animate-spin text-[#1f3a60]" />
                                        </span>
                                    ) : null}
                                </div>

                                <div className="pointer-events-none flex flex-1 flex-col gap-0.5 px-2.5 pb-1 pt-2">
                                    {/* Две строки, а не одна: заголовок темы редко
                                        помещается в строку узкой карточки, а обрезка по
                                        первой строке съедала бы как раз ту часть, которая
                                        отличает колоду от соседней. Полный текст остаётся
                                        в подсказке. min-h держит карточки в ряду одной
                                        высоты, когда у одной заголовок в строку, а у
                                        другой в две. */}
                                    <p
                                        className={cn(
                                            'line-clamp-2 min-h-[2.1rem] break-words text-[13px] font-semibold leading-[1.3]',
                                            presentation.title ? 'text-slate-900' : 'text-slate-400',
                                        )}
                                        title={cardName}
                                    >
                                        {cardName}
                                    </p>

                                    <p className="truncate text-[11px] text-slate-400">
                                        {t('presentations.cardMeta', {
                                            date: formatLocaleDate(presentation.created_at, locale, {
                                                day: 'numeric',
                                                month: 'short',
                                                year: 'numeric',
                                            }, '—'),
                                            slides: presentation.slide_count,
                                            language: String(presentation.language || '').toUpperCase(),
                                        })}
                                    </p>

                                    {isInProgress ? (
                                        /* Минуты ожидания надо назвать заранее: без этой
                                           строки «Генерируется 0%» через полминуты читается
                                           как «зависло». */
                                        <p className="mt-0.5 line-clamp-3 text-[11px] leading-4 text-slate-400">
                                            {t('presentations.inProgressHint')}
                                        </p>
                                    ) : null}

                                    {errorMessage ? (
                                        /* error_text остаётся и подсказкой при наведении:
                                           в тексте он появляется только когда код неизвестен
                                           (см. resolvePresentationErrorMessage). Событиям
                                           мыши блок возвращён явно — иначе подсказки не было
                                           бы вовсе. */
                                        <p
                                            className="pointer-events-auto mt-1 flex items-start gap-1 rounded-lg bg-red-50 px-2 py-1.5 text-[11px] leading-4 text-red-600"
                                            title={presentation.error_text || errorMessage}
                                        >
                                            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                                            <span className="line-clamp-3 min-w-0 break-words">{errorMessage}</span>
                                        </p>
                                    ) : null}
                                </div>

                                {/* Действия — значками без подписей, но кнопками 40x40
                                    (size="icon"): на карточке шириной в шестую часть ряда
                                    текстовые кнопки не помещаются, а мельчить площадь
                                    нажатия нельзя. Название действия не потеряно — оно в
                                    title и в aria-label, то есть доступно и наведению, и
                                    чтению с экрана. */}
                                <div className="relative z-10 flex items-center justify-end gap-0.5 px-1.5 pb-1.5">
                                    {status === 'error' && onReorder ? (
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="icon"
                                            className="mr-auto"
                                            disabled={reorderBlocked}
                                            title={reorderBlocked && reorderingId !== presentation.id
                                                ? t('presentations.orderAgainDisabledInProgress')
                                                : t('presentations.orderAgain')}
                                            aria-label={t('presentations.orderAgain')}
                                            isLoading={reorderingId === presentation.id}
                                            onClick={() => handleReorder(presentation)}
                                        >
                                            {reorderingId === presentation.id ? null : <RefreshCw className="h-4 w-4" />}
                                        </Button>
                                    ) : null}

                                    {/* Скачивание только у готовой: у остальных файла на
                                        диске ещё нет, и кнопка вела бы в 409/404. */}
                                    {isReady ? (
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => handleDownload(presentation)}
                                            isLoading={downloadingId === presentation.id}
                                            title={t('presentations.download')}
                                            aria-label={t('presentations.download')}
                                        >
                                            {downloadingId === presentation.id ? null : <Download className="h-4 w-4" />}
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
                                        size="icon"
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
                                    </Button>
                                </div>
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
                                    {/* Без внутреннего номера презентации: диалог открыт
                                        из конкретной карточки, а машинный ID подтвердить
                                        удаление не помогает — он ничего не значит для
                                        того, кто заказывал колоду. */}
                                    {/* Подпись — та же, что на карточке, и берётся тем же
                                        resolveCardName: в подтверждении удаления обязано
                                        стоять ровно то имя, по которому пользователь эту
                                        колоду и опознал. */}
                                    {t('presentations.deleteConfirmDescription', {
                                        name: resolveCardName(deleteTarget),
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
        </div>
    );
};

export default PresentationList;
