import React, { useState } from 'react';
import { AlertTriangle, ChevronDown, Presentation, RefreshCw, Sparkles } from 'lucide-react';

import PresentationTemplateGallery from './PresentationTemplateGallery';
import { Button } from '../ui/Button';
import Input from '../ui/Input';
import { cn } from '../../lib/utils';
import { useLocale } from '../../i18n';
import {
    DEFAULT_SLIDE_COUNT,
    DESCRIPTION_MAX,
    PRESENTATION_LANGUAGES,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    localeToPresentationLanguage,
    resolveTemplateName,
} from '../../lib/presentations';

const LANGUAGE_LABEL_KEYS = {
    ru: 'presentations.languageRu',
    tj: 'presentations.languageTj',
};

/**
 * Заказ презентации.
 *
 * Главное действие здесь — получить колоду, а не настроить её: у всех четырёх
 * параметров есть рабочее умолчание (шаблон предвыбран, язык взят из локали,
 * слайдов DEFAULT_SLIDE_COUNT, описание необязательно), поэтому достаточно
 * нажать одну кнопку. Раньше об этом нельзя было догадаться: четыре поля с
 * подсказками занимали весь экран и требовали решений, которых никто не ждал.
 * Теперь параметры лежат под раскрытием, а их применённое состояние выписано
 * строкой рядом с кнопкой — свёрнутая панель не должна означать «неизвестно
 * что закажут».
 *
 * Проверки полей стоят не «вместо» серверных, а до них: сервер всё равно
 * отвергнет число слайдов вне диапазона (presentation.value_out_of_range) и
 * слишком длинное описание (presentation.description_too_long), но узнавать об
 * этом из отказа — значит терять заполненную форму на очевидной опечатке.
 * Границы берутся из lib/presentations.js — зеркала серверных констант — и
 * показаны у самого поля: их видно до отправки, то есть в тот момент, когда
 * число вообще можно ввести.
 */
const PresentationForm = ({
    templates,
    templatesLoading,
    templatesError,
    onReloadTemplates,
    onSubmit,
    isSubmitting,
    submitError,
}) => {
    const { locale, t } = useLocale();

    const [templateKey, setTemplateKey] = useState('');
    // Язык колоды по умолчанию — язык интерфейса: чаще всего заказывают на том,
    // на котором работают. Дальше значение живёт своей жизнью и переключением
    // локали не сбрасывается — иначе выбранный вручную язык молча менялся бы.
    const [language, setLanguage] = useState(() => localeToPresentationLanguage(locale));
    const [slideCount, setSlideCount] = useState(String(DEFAULT_SLIDE_COUNT));
    const [description, setDescription] = useState('');
    const [validationError, setValidationError] = useState('');

    const [optionsOpen, setOptionsOpen] = useState(false);
    /**
     * Панель монтируется при первом раскрытии и дальше остаётся в дереве
     * скрытой атрибутом hidden.
     *
     * Так превью шаблонов (каждое — отдельный запрос за картинкой) не грузятся
     * у того, кто заказывает колоду с умолчаниями, а тот, кто панель открывал,
     * не платит за перезагрузку превью на каждое сворачивание.
     */
    const [optionsMounted, setOptionsMounted] = useState(false);

    const toggleOptions = () => {
        setOptionsMounted(true);
        setOptionsOpen((previous) => !previous);
    };

    /**
     * Действующий выбор шаблона ВЫЧИСЛЯЕТСЯ, а не досылается в состояние
     * эффектом после загрузки списка.
     *
     * Разница не в стиле: эффект, подставляющий первый шаблон, — это лишний
     * прогон рендера на каждый ответ сервера, и он же оставляет окно, в котором
     * форма считает выбранным шаблон, которого в списке уже нет (шаблон убрали
     * релизом). Здесь это одно выражение: выбор пользователя, пока он есть в
     * списке, иначе первый доступный.
     */
    const selectedTemplateKey = templates.some((template) => template.key === templateKey)
        ? templateKey
        : (templates[0]?.key || '');

    /**
     * Что именно закажут — одной строкой на языке пользователя.
     *
     * Части, которых сейчас нет (шаблоны ещё грузятся, поле числа очищено,
     * описание не заполнено), выпадают, а не показываются пустыми: строка
     * описывает состояние заказа, и придумывать ей недостающее нельзя.
     */
    const summary = [
        resolveTemplateName(templates.find((template) => template.key === selectedTemplateKey), locale),
        t(LANGUAGE_LABEL_KEYS[language]),
        slideCount.trim() ? t('presentations.optionsSlides', { count: slideCount.trim() }) : '',
        description.trim() ? t('presentations.optionsDescription') : '',
    ].filter(Boolean).join(' · ');

    const handleSubmit = (event) => {
        event.preventDefault();
        if (isSubmitting || !selectedTemplateKey) return;

        // Отказ проверки раскрывает панель: поле, о котором говорит сообщение,
        // иначе осталось бы скрытым, и исправить его было бы негде.
        const reject = (message) => {
            setValidationError(message);
            setOptionsMounted(true);
            setOptionsOpen(true);
        };

        const parsedSlideCount = Number(slideCount);
        if (!Number.isInteger(parsedSlideCount)) {
            reject(t('presentations.slideCountInvalid'));
            return;
        }

        if (parsedSlideCount < SLIDE_COUNT_MIN || parsedSlideCount > SLIDE_COUNT_MAX) {
            reject(t('presentations.slideCountRange', { min: SLIDE_COUNT_MIN, max: SLIDE_COUNT_MAX }));
            return;
        }

        const trimmedDescription = description.trim();
        if (trimmedDescription.length > DESCRIPTION_MAX) {
            reject(t('presentations.descriptionTooLong', { max: DESCRIPTION_MAX }));
            return;
        }

        setValidationError('');
        onSubmit({
            template_key: selectedTemplateKey,
            language,
            slide_count: parsedSlideCount,
            // Описание необязательно: пустая строка — законный заказ «просто
            // соберите колоду по блокноту», и подставлять вместо неё что-либо
            // от себя нельзя.
            description: trimmedDescription,
        });
    };

    return (
        <form onSubmit={handleSubmit} className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5 flex items-center gap-3">
                <div className="rounded-xl bg-[#1f3a60]/10 p-2 text-[#1f3a60]">
                    <Presentation className="h-5 w-5" />
                </div>
                <div>
                    <h2 className="text-lg font-semibold text-slate-900">{t('presentations.formTitle')}</h2>
                    <p className="text-sm text-slate-500">{t('presentations.formDescription')}</p>
                </div>
            </div>

            <div className="space-y-5">
                {/* Почему заказ сейчас невозможен — НАД раскрытием, а не внутри
                    него: свёрнутая панель спрятала бы и причину, и «Повторить»,
                    а кнопка заказа при этом стоит выключенной без объяснения.
                    Внутри галереи этих же состояний больше нет — один экземпляр
                    сообщения, а не два расходящихся. */}
                {templatesError ? (
                    <div role="alert" className="flex flex-col gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                        <span className="flex items-center gap-2 font-semibold">
                            <AlertTriangle className="h-4 w-4" />
                            {templatesError}
                        </span>
                        {onReloadTemplates ? (
                            <Button type="button" variant="outline" size="sm" className="self-start" onClick={onReloadTemplates}>
                                <RefreshCw className="h-4 w-4" />
                                {t('presentations.retry')}
                            </Button>
                        ) : null}
                    </div>
                ) : null}

                {!templatesError && !templatesLoading && templates.length === 0 ? (
                    <p className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
                        {t('presentations.templatesEmpty')}
                    </p>
                ) : null}

                <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                    <button
                        type="button"
                        onClick={toggleOptions}
                        aria-expanded={optionsOpen}
                        aria-controls="presentation-options"
                        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1"
                    >
                        <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', optionsOpen && 'rotate-180')} />
                        {t('presentations.options')}
                    </button>

                    {/* Раскрытая панель показывает то же самое самими полями —
                        повторять строку рядом значит спорить сам с собой. */}
                    {!optionsOpen ? (
                        <span className="min-w-0 break-words text-xs text-slate-500">{summary}</span>
                    ) : null}
                </div>

                {optionsMounted ? (
                    <div id="presentation-options" hidden={!optionsOpen} className="space-y-5">
                        <div>
                            <p id="presentation-template-label" className="mb-2 text-sm font-semibold text-slate-700">
                                {t('presentations.templateLabel')}
                            </p>
                            <PresentationTemplateGallery
                                templates={templates}
                                selectedKey={selectedTemplateKey}
                                onSelect={setTemplateKey}
                                isLoading={templatesLoading}
                                disabled={isSubmitting}
                                labelId="presentation-template-label"
                            />
                        </div>

                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                            <div>
                                <label htmlFor="presentation-language" className="mb-2 block text-sm font-semibold text-slate-700">
                                    {t('presentations.languageLabel')}
                                </label>
                                <select
                                    id="presentation-language"
                                    value={language}
                                    onChange={(event) => setLanguage(event.target.value)}
                                    disabled={isSubmitting}
                                    className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 shadow-sm transition-colors focus-visible:border-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {PRESENTATION_LANGUAGES.map((code) => (
                                        <option key={code} value={code}>{t(LANGUAGE_LABEL_KEYS[code])}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label htmlFor="presentation-slide-count" className="mb-2 block text-sm font-semibold text-slate-700">
                                    {t('presentations.slideCountLabel')}
                                </label>
                                <Input
                                    id="presentation-slide-count"
                                    type="number"
                                    inputMode="numeric"
                                    min={SLIDE_COUNT_MIN}
                                    max={SLIDE_COUNT_MAX}
                                    step={1}
                                    value={slideCount}
                                    onChange={(event) => setSlideCount(event.target.value)}
                                    disabled={isSubmitting}
                                    aria-describedby="presentation-slide-count-hint"
                                />
                                {/* Границы диапазона — зеркало серверной проверки, и
                                    видны они там, где число вводят: заказ, упавший с
                                    presentation.value_out_of_range уже после отправки,
                                    стоит пользователю минут ожидания. */}
                                <p id="presentation-slide-count-hint" className="mt-1.5 text-xs text-slate-500">
                                    {t('presentations.slideCountHint', { min: SLIDE_COUNT_MIN, max: SLIDE_COUNT_MAX })}
                                </p>
                            </div>
                        </div>

                        <div>
                            <label htmlFor="presentation-description" className="mb-2 block text-sm font-semibold text-slate-700">
                                {t('presentations.descriptionLabel')}
                            </label>
                            <textarea
                                id="presentation-description"
                                value={description}
                                onChange={(event) => setDescription(event.target.value)}
                                placeholder={t('presentations.descriptionPlaceholder')}
                                maxLength={DESCRIPTION_MAX}
                                rows={4}
                                disabled={isSubmitting}
                                aria-describedby="presentation-description-counter"
                                className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm transition-colors placeholder:text-slate-400 focus-visible:border-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30 disabled:cursor-not-allowed disabled:opacity-50"
                            />
                            {/* Счётчик без aria-live: поле упирается в maxLength молча, и
                                пояснение нужно рядом, а не объявлением на каждый символ.
                                Связан с полем через aria-describedby — так его читают
                                один раз, при входе в поле. */}
                            <p id="presentation-description-counter" className="mt-1.5 text-xs text-slate-500">
                                {t('presentations.descriptionCounter', { count: description.length, max: DESCRIPTION_MAX })}
                            </p>
                        </div>
                    </div>
                ) : null}

                {validationError || submitError ? (
                    <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                        {validationError || submitError}
                    </p>
                ) : null}

                <div className="flex justify-end">
                    <Button
                        type="submit"
                        size="lg"
                        className="w-full sm:w-auto"
                        isLoading={isSubmitting}
                        disabled={!selectedTemplateKey}
                        title={!selectedTemplateKey ? t('presentations.submitDisabledHint') : undefined}
                    >
                        <Sparkles className="h-4 w-4" />
                        {isSubmitting ? t('presentations.submitting') : t('presentations.submit')}
                    </Button>
                </div>
            </div>
        </form>
    );
};

export default PresentationForm;
