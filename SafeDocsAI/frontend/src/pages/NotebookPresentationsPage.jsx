import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';

import PresentationForm from '../components/presentations/PresentationForm';
import PresentationList from '../components/presentations/PresentationList';
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
    useIndexingPoll(hasPresentationsInProgress(items), pollPresentations);

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

            const position = resolveQueuePosition(created);
            setCreatedMessage(position != null
                ? t('presentations.createdQueuedPosition', { position })
                : t('presentations.createdQueued'));
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

    return (
        // Раздел прокручивается сам, внутри отведённой блокнотом высоты: полосу
        // вкладок над содержимым уводить из виду при длинной истории заказов
        // незачем.
        <div className="scrollbar-soft h-full min-h-0 space-y-6 overflow-y-auto pr-1">
            <PresentationForm
                templates={templates}
                templatesLoading={templatesLoading}
                templatesError={templatesError}
                onReloadTemplates={loadTemplates}
                onSubmit={handleCreate}
                isSubmitting={isCreating}
                submitError={createError}
            />

            {/* Результат отправки объявляется отдельно от формы: пользователь,
                работающий с клавиатуры, остаётся на кнопке и иначе не узнаёт,
                что заказ принят. */}
            <div role="status" aria-live="polite">
                {createdMessage ? (
                    <p className="rounded-2xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
                        {createdMessage}
                    </p>
                ) : null}
            </div>

            <PresentationList
                items={items}
                total={total}
                templates={templates}
                isLoading={isListLoading}
                error={listError}
                onRetry={handleRetry}
                onDeleted={handleDeleted}
            />
        </div>
    );
};

export default NotebookPresentationsPage;
