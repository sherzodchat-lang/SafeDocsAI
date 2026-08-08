import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Plus } from 'lucide-react';

import PresentationForm from '../components/presentations/PresentationForm';
import PresentationList from '../components/presentations/PresentationList';
import { Button } from '../components/ui/Button';
import { useIndexingPoll } from '../hooks/useIndexingPoll';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';
import { hasPresentationsInProgress, resolveQueuePosition } from '../lib/presentations';
// Разбор постраничного ответа общий на весь проект: тело — голый массив, общее
// число — заголовком X-Total-Count. Имя функции историческое (первым таким
// списком были источники), но разбирает она форму ответа, а не источники.
import { normalizeSourcesResponse } from '../lib/sources';
import { presentationsService } from '../services/presentationsService';

/**
 * Сколько презентаций читаем за раз.
 *
 * Свежие сверху, и заказывают их поштучно, поэтому долистывать здесь нечего:
 * одна страница закрывает обозримую историю блокнота. Если презентаций больше,
 * счётчик над списком честно говорит, сколько показано из скольких, — молчать
 * об остатке хуже, чем не показать его.
 */
const LIST_PAGE_LIMIT = 50;

/**
 * Потолок числа попыток опроса — свой, а не умолчание useIndexingPoll (~10
 * минут).
 *
 * Умолчание хука посчитано под индексацию источника, а генерация колоды дольше
 * по построению: замер приёмки — 444 секунды на 15 слайдов, и это ПОСЛЕ
 * очереди, которая общая на систему и берёт задачи по одной, так что перед
 * заказом может стоять несколько чужих. С умолчанием опрос умирал посреди
 * здоровой генерации, карточка застывала на последнем проценте, и возобновить
 * наблюдение было нечем, кроме перезагрузки страницы.
 *
 * 600 попыток по 4 секунды — 40 минут видимой вкладки (скрытая попытки не
 * тратит): несколько чужих колод в очереди плюс своя с запасом. Роль потолка
 * та же, что у умолчания, — не дёргать сервер вечно из-за задачи, застрявшей
 * навсегда, — и совсем снимать его поэтому нельзя.
 */
const POLL_MAX_ATTEMPTS = 600;

/**
 * Раздел презентаций блокнота: заказ и история заказов.
 *
 * Роль проверяется маршрутом (RequireContentAccess в App.jsx), поэтому здесь её
 * снова не спрашиваем: два места, решающих один вопрос, рано или поздно
 * разойдутся. Настоящая граница прав всё равно на сервере — он отвечает
 * presentation.role_not_allowed независимо от того, что решил браузер.
 */
const NotebookPresentationsPage = () => {
    const { notebookId } = useParams();
    const { t } = useLocale();

    const [templates, setTemplates] = useState([]);
    const [templatesLoading, setTemplatesLoading] = useState(true);
    const [templatesError, setTemplatesError] = useState('');

    const [items, setItems] = useState([]);
    const [total, setTotal] = useState(null);
    const [isListLoading, setIsListLoading] = useState(true);
    // Ошибка загрузки списка держится отдельным состоянием и НЕ трогает items:
    // неудачное обновление не должно превращать показанный список в пустоту.
    const [listError, setListError] = useState('');

    const [isCreating, setIsCreating] = useState(false);
    const [createError, setCreateError] = useState('');
    const [createdMessage, setCreatedMessage] = useState('');

    /**
     * Окно заказа. Раньше его содержимое стояло на странице постоянно и
     * занимало пол-экрана над списком колод; теперь наверху одна кнопка, а всё
     * остальное открывается по требованию (см. PresentationForm).
     */
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    /**
     * Побеждает последний ЗАПРОШЕННЫЙ, а не последний ответивший.
     *
     * Тот же приём, что в SourcesContext: у каждого запроса свой номер, и ответ
     * с чужим номером выбрасывается. Без этого фоновый опрос, отправленный
     * раньше и вернувшийся позже ручного «Повторить», затирал бы свежий список
     * устаревшим — а на глазах у пользователя это выглядит как самопроизвольный
     * откат прогресса.
     *
     * Счётчик увеличивается и при уходе со страницы (cleanup эффекта ниже):
     * запрос, оставшийся в полёте, вернётся с устаревшим номером, и его ответ
     * не попадёт в состояние размонтированного экрана.
     */
    const lastRequestIdRef = useRef(0);

    // Куда прокручивать после принятого заказа — к сообщению «заказ принят».
    // Нужно это и заказу из окна (оно закрывается, и подтверждение остаётся
    // единственным следом сделанного), и повторному заказу из сетки: карточка
    // «Заказать заново» может стоять глубоко в истории, и без прокрутки
    // и подтверждение, и новая карточка «В очереди» появлялись бы за верхней
    // границей экрана — то есть молча.
    const createdMessageRef = useRef(null);

    const loadTemplates = useCallback(async () => {
        try {
            setTemplatesLoading(true);
            setTemplatesError('');
            const response = await presentationsService.getTemplates();
            setTemplates(Array.isArray(response.data) ? response.data : []);
        } catch (error) {
            console.error('Failed to fetch presentation templates:', error);
            setTemplates([]);
            setTemplatesError(resolveApiErrorMessage(error, t, 'presentations.templatesLoadFailed'));
        } finally {
            setTemplatesLoading(false);
        }
    }, [t]);

    const loadList = useCallback(async ({ background = false } = {}) => {
        if (!notebookId) return;

        const requestId = lastRequestIdRef.current + 1;
        lastRequestIdRef.current = requestId;

        if (!background) {
            setIsListLoading(true);
            setListError('');
        }

        try {
            const response = await presentationsService.getPage(notebookId, { skip: 0, limit: LIST_PAGE_LIMIT });
            if (lastRequestIdRef.current !== requestId) return;

            const { items: loaded, total: loadedTotal } = normalizeSourcesResponse(response);
            setItems(loaded);
            setTotal(loadedTotal != null ? loadedTotal : loaded.length);
            setListError('');
        } catch (error) {
            console.error('Failed to fetch presentations:', error);
            if (lastRequestIdRef.current !== requestId) return;

            // Сбой ФОНОВОГО тика не показываем вовсе: список на экране остаётся
            // верным, а сеть моргнула на одну секунду — сообщение об ошибке в
            // этом месте только пугает. Следующий тик всё равно повторит запрос.
            if (background) return;

            setListError(resolveApiErrorMessage(error, t, 'presentations.loadFailed'));
        } finally {
            if (lastRequestIdRef.current === requestId && !background) {
                setIsListLoading(false);
            }
        }
    }, [notebookId, t]);

    useEffect(() => {
        loadTemplates();
    }, [loadTemplates]);

    useEffect(() => {
        loadList();

        return () => {
            // Уход со страницы (и смена блокнота) обесценивает запрос в полёте —
            // см. пояснение к lastRequestIdRef.
            lastRequestIdRef.current += 1;
        };
    }, [loadList]);

    // Смена блокнота: показанное принадлежит прошлому, и до первого ответа его
    // держать нельзя — иначе рядом с новым блокнотом висит чужая история.
    useEffect(() => {
        setItems([]);
        setTotal(null);
        setCreatedMessage('');
        setCreateError('');
        // Окно заказа принадлежит блокноту, из которого его открыли: оставить
        // его над другим блокнотом значит предложить заказать колоду не там.
        setIsDialogOpen(false);
    }, [notebookId]);

    const pollPresentations = useCallback(async () => {
        await loadList({ background: true });
    }, [loadList]);

    /**
     * Опрос идёт, пока хоть одна презентация в очереди или генерируется, и
     * прекращается сам, как только все дошли до ready/error.
     *
     * Механику даёт useIndexingPoll — тот же хук, что у индексации источников:
     * следующий тик ставится только после завершения предыдущего (setTimeout, а
     * не setInterval), в скрытой вкладке сервер не дёргается и попытка не
     * тратится, и есть потолок попыток (~10 минут) на случай задачи, застрявшей
     * в очереди навсегда. При уходе со страницы эффект хука снимает таймер, а
     * ответ уже отправленного запроса отсекается по lastRequestIdRef.
     */
    useIndexingPoll(hasPresentationsInProgress(items), pollPresentations, { maxAttempts: POLL_MAX_ATTEMPTS });

    const handleCreate = useCallback(async (payload) => {
        if (!notebookId || isCreating) return;

        try {
            setIsCreating(true);
            setCreateError('');
            setCreatedMessage('');
            const response = await presentationsService.create(notebookId, payload);
            const created = response.data;

            if (created?.id != null) {
                // Ответ 202 — это уже полноценный объект презентации: показываем
                // его сразу, не дожидаясь следующего тика опроса.
                setItems((previous) => (
                    previous.some((item) => item.id === created.id) ? previous : [created, ...previous]
                ));
                setTotal((previous) => (previous == null ? null : previous + 1));
            }

            // Заказ принят — окну больше нечего показывать, и закрывает его
            // именно УСПЕХ: при отказе окно остаётся открытым вместе с
            // набранными параметрами, иначе описание пришлось бы вводить заново
            // из-за одной неудачной отправки.
            setIsDialogOpen(false);

            const position = resolveQueuePosition(created);
            setCreatedMessage(position != null
                ? t('presentations.createdQueuedPosition', { position })
                : t('presentations.createdQueued'));

            // Прокрутка после отрисовки сообщения (см. createdMessageRef), и
            // 'nearest', а не 'start': если сообщение уже на экране, дёргать
            // страницу незачем.
            requestAnimationFrame(() => {
                createdMessageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        } catch (error) {
            console.error('Failed to create presentation:', error);
            setCreateError(resolveApiErrorMessage(error, t, 'presentations.createFailed'));
        } finally {
            setIsCreating(false);
        }
    }, [isCreating, notebookId, t]);

    // Кнопка «Повторить» зовёт загрузку без аргументов: событие клика, попав в
    // параметры loadList, притворилось бы объектом настроек.
    const handleRetry = useCallback(() => loadList(), [loadList]);

    const handleDeleted = useCallback((presentationId) => {
        setItems((previous) => previous.filter((item) => item.id !== presentationId));
        setTotal((previous) => (previous == null ? null : Math.max(0, previous - 1)));
    }, []);

    const openDialog = useCallback(() => {
        // Прошлое подтверждение при открытии окна убирается: «заказ принят»
        // относилось к предыдущей колоде, а рядом с новым заказом оно читается
        // как ответ на него.
        setCreatedMessage('');
        setCreateError('');
        setIsDialogOpen(true);
    }, []);

    const closeDialog = useCallback(() => setIsDialogOpen(false), []);

    /**
     * Пока в блокноте есть незавершённый заказ, новый ставить некуда: сервер
     * отвечает 409 presentation.generation_in_progress (одна активная колода на
     * блокнот — инвариант базы, а не только проверка обработчика). Поэтому
     * кнопка выключена ЗАРАНЕЕ и объясняет причину: провести пользователя через
     * два шага окна ради отказа на «Сгенерировать» — худшее из возможных
     * поведений.
     */
    const createBlocked = hasPresentationsInProgress(items);

    return (
        // Раздел прокручивается сам, внутри отведённой блокнотом высоты: полосу
        // вкладок над содержимым уводить из виду при длинной истории заказов
        // незачем.
        <div className="scrollbar-soft h-full min-h-0 space-y-5 overflow-y-auto pr-1">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                    <h2 className="text-lg font-semibold text-slate-900">{t('presentations.listTitle')}</h2>
                    {items.length > 0 ? (
                        <p className="mt-0.5 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
                            {t('presentations.listCount', { count: items.length, total: total ?? items.length })}
                        </p>
                    ) : null}
                </div>

                {/* Одна заметная кнопка вместо прежней формы на пол-экрана.
                    Всё, что нужно заказу, живёт в окне (PresentationForm). */}
                <Button
                    type="button"
                    size="lg"
                    onClick={openDialog}
                    disabled={createBlocked}
                    title={createBlocked ? t('presentations.createDisabledInProgress') : undefined}
                >
                    <Plus className="h-4 w-4" />
                    {t('presentations.create')}
                </Button>
            </div>

            {/* Результат отправки объявляется отдельно от окна — окно к этому
                моменту уже закрыто, — и живой областью: пользователь,
                работающий с клавиатуры, иначе не узнает, что заказ принят. */}
            <div ref={createdMessageRef} role="status" aria-live="polite">
                {createdMessage ? (
                    <p className="rounded-2xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
                        {createdMessage}
                    </p>
                ) : null}
            </div>

            {/* Отказ заказа показывает окно, пока оно открыто. Здесь он нужен
                для второго пути постановки в очередь — «Заказать заново» с
                карточки: окна при нём нет, и сообщению больше негде появиться. */}
            {createError && !isDialogOpen ? (
                <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                    {createError}
                </p>
            ) : null}

            <PresentationList
                items={items}
                templates={templates}
                isLoading={isListLoading}
                error={listError}
                onRetry={handleRetry}
                onDeleted={handleDeleted}
                // Повторный заказ из сетки идёт тем же путём, что заказ из
                // окна: одна точка постановки в очередь, одно место для
                // ошибок и подтверждения.
                onReorder={handleCreate}
                onCreate={openDialog}
            />

            <PresentationForm
                isOpen={isDialogOpen}
                onClose={closeDialog}
                templates={templates}
                templatesLoading={templatesLoading}
                templatesError={templatesError}
                onReloadTemplates={loadTemplates}
                onSubmit={handleCreate}
                isSubmitting={isCreating}
                submitError={createError}
            />
        </div>
    );
};

export default NotebookPresentationsPage;
