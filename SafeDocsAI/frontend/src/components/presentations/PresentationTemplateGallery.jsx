import React, { useEffect, useRef, useState } from 'react';
import { Check, ImageOff, Loader2 } from 'lucide-react';

import { cn } from '../../lib/utils';
import { useLocale } from '../../i18n';
import { resolveTemplateName } from '../../lib/presentations';
import { presentationsService } from '../../services/presentationsService';

/**
 * Превью шаблона.
 *
 * Картинка тянется отдельным запросом по preview_url из ответа (см.
 * presentationsService.getTemplatePreviewBlob) и живёт как object URL, который
 * обязательно освобождается: галерея перерисовывается при каждом обновлении
 * списка шаблонов, и незакрытые blob'ы копились бы до перезагрузки страницы.
 *
 * Сбой картинки не прячет шаблон: выбирают его по имени, а превью — подсказка.
 * Поэтому вместо карточки показывается заглушка, а не пустое место.
 */
const TemplatePreview = ({ previewUrl, alt }) => {
    const { t } = useLocale();
    // Результат помнится ВМЕСТЕ с адресом, за которым ходили. Так состояние
    // сбрасывается само при смене шаблона — без гашения его прямо в теле
    // эффекта, то есть без лишнего каскада перерисовок.
    const [result, setResult] = useState({ url: '', source: '', failed: false });

    useEffect(() => {
        if (!previewUrl) return undefined;

        let active = true;
        let objectUrl = '';

        presentationsService.getTemplatePreviewBlob(previewUrl)
            .then((response) => {
                if (!active) return;
                objectUrl = URL.createObjectURL(response.data);
                setResult({ url: previewUrl, source: objectUrl, failed: false });
            })
            .catch((error) => {
                console.error('Failed to fetch presentation template preview:', error);
                if (active) setResult({ url: previewUrl, source: '', failed: true });
            });

        return () => {
            active = false;
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        };
    }, [previewUrl]);

    const isCurrent = Boolean(previewUrl) && result.url === previewUrl;
    // Шаблон без preview_url — это не сбой запроса, а отсутствие картинки;
    // показывается та же заглушка.
    const failed = !previewUrl || (isCurrent && result.failed);
    const source = isCurrent ? result.source : '';

    if (failed) {
        return (
            <div className="flex h-32 w-full flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-slate-200 bg-slate-50 text-slate-400">
                <ImageOff className="h-5 w-5" />
                <span className="px-3 text-center text-[11px] leading-4">{t('presentations.previewUnavailable')}</span>
            </div>
        );
    }

    if (!source) {
        return (
            <div className="flex h-32 w-full items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-400">
                <Loader2 className="h-5 w-5 animate-spin" />
            </div>
        );
    }

    return (
        <img
            src={source}
            alt={alt}
            className="h-32 w-full rounded-xl border border-slate-200 bg-white object-contain"
        />
    );
};

/**
 * Галерея шаблонов — НАСТОЯЩАЯ радиогруппа, а не набор кликабельных карточек.
 *
 * Что это значит на практике и почему сделано именно так:
 *   * контейнер объявлен role="radiogroup" и подписан заголовком поля, поэтому
 *     программа чтения с экрана называет группу целиком, а не читает подряд
 *     четыре безымянные кнопки;
 *   * карточки — role="radio" с aria-checked, то есть выбранность объявляется,
 *     а не остаётся видимой только зрячему по цвету рамки;
 *   * фокус по группе один (roving tabindex): Tab заводит в неё и уводит
 *     дальше, а выбор внутри делается стрелками — так ведут себя нативные
 *     радиокнопки, и любое другое поведение здесь ощущается сломанным;
 *   * Space/Enter выбирают карточку под фокусом: <button> активируется ими сам,
 *     отдельный обработчик не нужен и только разошёлся бы с нативным.
 */
const PresentationTemplateGallery = ({
    templates,
    selectedKey,
    onSelect,
    isLoading,
    disabled = false,
    labelId,
}) => {
    const { locale, t } = useLocale();
    const itemRefs = useRef([]);

    const focusItem = (index) => {
        const target = itemRefs.current[index];
        if (!target) return;
        target.focus();
        onSelect(templates[index].key);
    };

    const handleKeyDown = (event, index) => {
        if (disabled || templates.length === 0) return;

        const forward = event.key === 'ArrowRight' || event.key === 'ArrowDown';
        const backward = event.key === 'ArrowLeft' || event.key === 'ArrowUp';

        if (forward || backward) {
            event.preventDefault();
            const shift = forward ? 1 : -1;
            // Перенос по кругу: у нативной радиогруппы стрелка с последнего
            // элемента возвращает на первый, и обрывать её здесь незачем.
            focusItem((index + shift + templates.length) % templates.length);
            return;
        }

        if (event.key === 'Home') {
            event.preventDefault();
            focusItem(0);
            return;
        }

        if (event.key === 'End') {
            event.preventDefault();
            focusItem(templates.length - 1);
        }
    };

    if (isLoading && templates.length === 0) {
        return (
            <div role="status" className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                {t('presentations.templatesLoading')}
            </div>
        );
    }

    // Пустой список и сбой загрузки объявляет форма, над раскрытием: причина,
    // по которой заказ невозможен, не должна прятаться вместе с панелью.
    if (templates.length === 0) return null;

    // Точка входа фокуса: выбранная карточка, а до выбора — первая. Без этого
    // Tab либо проваливался бы мимо группы, либо обходил все карточки подряд.
    const focusableIndex = Math.max(0, templates.findIndex((template) => template.key === selectedKey));

    return (
        <div
            role="radiogroup"
            aria-labelledby={labelId}
            aria-required="true"
            className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
        >
            {templates.map((template, index) => {
                const isSelected = template.key === selectedKey;
                const name = resolveTemplateName(template, locale) || template.key;

                return (
                    <button
                        key={template.key}
                        ref={(element) => { itemRefs.current[index] = element; }}
                        type="button"
                        role="radio"
                        aria-checked={isSelected}
                        tabIndex={index === focusableIndex ? 0 : -1}
                        disabled={disabled}
                        onClick={() => onSelect(template.key)}
                        onKeyDown={(event) => handleKeyDown(event, index)}
                        className={cn(
                            'flex flex-col gap-3 rounded-2xl border p-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-60',
                            isSelected
                                ? 'border-[#1f3a60] bg-[#1f3a60]/5 shadow-sm'
                                : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50',
                        )}
                    >
                        <TemplatePreview
                            previewUrl={template.preview_url}
                            alt={t('presentations.previewAlt', { name })}
                        />
                        <span className="flex items-center justify-between gap-2">
                            <span className="min-w-0 break-words text-sm font-semibold text-slate-900">{name}</span>
                            {/* Выбранность показана не только заливкой: цвет рамки
                                неразличим при монохромном зрении и на печати. */}
                            <span
                                aria-hidden="true"
                                className={cn(
                                    'flex h-5 w-5 shrink-0 items-center justify-center rounded-full border',
                                    isSelected ? 'border-[#1f3a60] bg-[#1f3a60] text-white' : 'border-slate-300 bg-white',
                                )}
                            >
                                {isSelected ? <Check className="h-3.5 w-3.5" /> : null}
                            </span>
                        </span>
                    </button>
                );
            })}
        </div>
    );
};

export default PresentationTemplateGallery;
