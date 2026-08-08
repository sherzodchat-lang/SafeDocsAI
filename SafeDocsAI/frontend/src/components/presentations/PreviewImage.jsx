import React, { useEffect, useState } from 'react';
import { ImageOff, Loader2 } from 'lucide-react';

import { cn } from '../../lib/utils';
import { useLocale } from '../../i18n';
import { presentationsService } from '../../services/presentationsService';

/**
 * Кэш картинок превью на весь сеанс: адрес -> обещание блоба.
 *
 * Нужен потому, что превью показывают два места сразу — галерея оформлений в
 * окне заказа (картинка ШАБЛОНА) и каждая карточка в сетке колод (первая
 * страница САМОЙ колоды). Без кэша четыре шаблона в галерее означали бы четыре
 * запроса при каждом открытии окна, а сетка из двадцати колод — двадцать
 * запросов при каждом тике опроса статуса.
 *
 * Хранится именно БЛОБ, а не object URL: URL освобождает тот, кто его создал
 * (см. эффект ниже), и общий на всех адрес первый же размонтированный компонент
 * отозвал бы из-под остальных — картинки в живых карточках стали бы битыми.
 *
 * Неудача не запоминается: сбой сети — состояние минуты, а не свойство адреса,
 * и следующий показ обязан сходить за картинкой заново. Для колод у этого есть
 * второй смысл: первый запрос картинки заставляет сервер её нарисовать, и
 * отказ по таймауту сети не должен навсегда оставить карточку без превью.
 */
const previewBlobCache = new Map();

const loadPreviewBlob = (previewUrl) => {
    const cached = previewBlobCache.get(previewUrl);
    if (cached) return cached;

    const pending = presentationsService.getPreviewBlob(previewUrl)
        .then((response) => response.data)
        .catch((error) => {
            previewBlobCache.delete(previewUrl);
            throw error;
        });

    previewBlobCache.set(previewUrl, pending);
    return pending;
};

/**
 * Картинка превью — одна на два места показа: шаблон в галерее и колода в сетке.
 *
 * Компонент не знает и не должен знать, ЧТО именно на картинке: он получает
 * адрес (preview_url из ответа сервера — у шаблона свой, у колоды свой) и
 * отвечает за одно — показать её, пока она грузится, и не сломать раскладку,
 * если она не пришла. Разделять его на два одинаковых было бы удвоением кода
 * загрузки, отзыва object URL и заглушек.
 *
 * Картинка тянется отдельным запросом (см. presentationsService.getPreviewBlob)
 * и живёт как object URL, который обязательно освобождается: и галерея, и сетка
 * перерисовываются на каждом обновлении данных, а незакрытые blob'ы копились бы
 * до перезагрузки страницы.
 *
 * Сбой картинки ничего не прячет: шаблон выбирают по имени, колоду узнают по
 * заголовку, а превью — подсказка. Поэтому вместо пустого места показывается
 * заглушка, которая ОБЪЯСНЯЕТ, что картинки нет, а не просто серый прямоугольник.
 *
 * Размер задаёт вызывающий (className), потому что рамки у мест показа разные:
 * в галерее это невысокая плитка в ряду, на карточке колоды — превью во всю её
 * ширину. Заглушки берут тот же класс, что и картинка, — иначе сетка прыгала
 * бы, когда одно превью загрузилось, а другое нет.
 */
export const PreviewImage = ({ previewUrl, alt, className, imageClassName }) => {
    const { t } = useLocale();
    // Результат помнится ВМЕСТЕ с адресом, за которым ходили. Так состояние
    // сбрасывается само при смене адреса — без гашения его прямо в теле
    // эффекта, то есть без лишнего каскада перерисовок.
    const [result, setResult] = useState({ url: '', source: '', failed: false });

    useEffect(() => {
        if (!previewUrl) return undefined;

        let active = true;
        let objectUrl = '';

        loadPreviewBlob(previewUrl)
            .then((blob) => {
                if (!active) return;
                objectUrl = URL.createObjectURL(blob);
                setResult({ url: previewUrl, source: objectUrl, failed: false });
            })
            .catch((error) => {
                console.error('Failed to fetch presentation preview:', error);
                if (active) setResult({ url: previewUrl, source: '', failed: true });
            });

        return () => {
            active = false;
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        };
    }, [previewUrl]);

    const isCurrent = Boolean(previewUrl) && result.url === previewUrl;
    // Пустой preview_url — это не сбой запроса, а честное «картинки не будет»
    // (шаблон без превью, колода, которая ещё не собралась); показывается та же
    // заглушка.
    const failed = !previewUrl || (isCurrent && result.failed);
    const source = isCurrent ? result.source : '';

    if (failed) {
        return (
            <div className={cn('flex flex-col items-center justify-center gap-1.5 bg-slate-50 text-slate-400', className)}>
                <ImageOff className="h-5 w-5" />
                <span className="px-2 text-center text-[10px] leading-4">{t('presentations.previewUnavailable')}</span>
            </div>
        );
    }

    if (!source) {
        return (
            <div role="status" className={cn('flex items-center justify-center bg-slate-50 text-slate-400', className)}>
                <Loader2 className="h-5 w-5 animate-spin" />
            </div>
        );
    }

    return <img src={source} alt={alt} className={cn('bg-white', className, imageClassName)} />;
};

export default PreviewImage;
