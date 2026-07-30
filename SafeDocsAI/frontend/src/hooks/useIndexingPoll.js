import { useEffect, useRef } from 'react';

const DEFAULT_INTERVAL_MS = 4000;
// Предохранитель: если источник завис в очереди, опрос всё равно остановится (~10 минут).
const DEFAULT_MAX_ATTEMPTS = 150;

/**
 * Периодически дёргает onPoll, пока shouldPoll === true.
 *
 * Индексация выполняется в фоновой очереди, поэтому статус источника меняется
 * уже после ответа сервера на загрузку: pending -> indexing -> indexed/error.
 *
 * onPoll получает { isActive } — проверку «эффект ещё жив», чтобы вызывающий код
 * не делал setState после размонтирования или после остановки опроса.
 */
export const useIndexingPoll = (shouldPoll, onPoll, options = {}) => {
    const { intervalMs = DEFAULT_INTERVAL_MS, maxAttempts = DEFAULT_MAX_ATTEMPTS } = options;

    // Колбэк держим в ref: его новая идентичность на каждом рендере не должна перезапускать таймер.
    const onPollRef = useRef(onPoll);

    useEffect(() => {
        onPollRef.current = onPoll;
    }, [onPoll]);

    useEffect(() => {
        if (!shouldPoll) return undefined;

        let active = true;
        let timerId = null;
        let attempts = 0;

        const isActive = () => active;

        function scheduleNext() {
            if (!active) return;
            timerId = setTimeout(tick, intervalMs);
        }

        // setTimeout, а не setInterval: следующий запрос ставим только после завершения предыдущего.
        async function tick() {
            if (!active) return;

            // В неактивной вкладке сервер не дёргаем и попытку не тратим.
            if (document.hidden) {
                scheduleNext();
                return;
            }

            attempts += 1;

            try {
                await onPollRef.current?.({ isActive });
            } catch (error) {
                console.error('Indexing poll failed:', error);
            }

            if (!active || attempts >= maxAttempts) return;
            scheduleNext();
        }

        scheduleNext();

        return () => {
            active = false;
            if (timerId) clearTimeout(timerId);
        };
    }, [shouldPoll, intervalMs, maxAttempts]);
};

export default useIndexingPoll;
