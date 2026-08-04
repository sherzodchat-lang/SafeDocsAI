import React, { useState } from 'react';
import { Presentation, Sparkles } from 'lucide-react';

import PresentationTemplateGallery from './PresentationTemplateGallery';
import { Button } from '../ui/Button';
import Input from '../ui/Input';
import { useLocale } from '../../i18n';
import {
    DEFAULT_SLIDE_COUNT,
    DESCRIPTION_MAX,
    PRESENTATION_LANGUAGES,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    localeToPresentationLanguage,
} from '../../lib/presentations';

const LANGUAGE_LABEL_KEYS = {
    ru: 'presentations.languageRu',
    tj: 'presentations.languageTj',
};

/**
 * Заказ презентации.
 *
 * Проверки полей стоят здесь не «вместо» серверных, а до них: сервер всё равно
 * отвергнет число слайдов вне диапазона (presentation.value_out_of_range) и
 * слишком длинное описание (presentation.description_too_long), но узнавать об
 * этом из отказа — значит терять заполненную форму на очевидной опечатке.
 * Границы берутся из lib/presentations.js — зеркала серверных констант.
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

    const handleSubmit = (event) => {
        event.preventDefault();
        if (isSubmitting || !selectedTemplateKey) return;

        const parsedSlideCount = Number(slideCount);
        if (!Number.isInteger(parsedSlideCount)) {
            setValidationError(t('presentations.slideCountInvalid'));
            return;
        }

        if (parsedSlideCount < SLIDE_COUNT_MIN || parsedSlideCount > SLIDE_COUNT_MAX) {
            setValidationError(t('presentations.slideCountRange', { min: SLIDE_COUNT_MIN, max: SLIDE_COUNT_MAX }));
            return;
        }

        const trimmedDescription = description.trim();
        if (trimmedDescription.length > DESCRIPTION_MAX) {
            setValidationError(t('presentations.descriptionTooLong', { max: DESCRIPTION_MAX }));
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
                <div>
                    <p id="presentation-template-label" className="mb-2 text-sm font-semibold text-slate-700">
                        {t('presentations.templateLabel')}
                    </p>
                    <p id="presentation-template-hint" className="mb-3 text-xs text-slate-500">
                        {t('presentations.templateHint')}
                    </p>
                    <PresentationTemplateGallery
                        templates={templates}
                        selectedKey={selectedTemplateKey}
                        onSelect={setTemplateKey}
                        isLoading={templatesLoading}
                        error={templatesError}
                        onRetry={onReloadTemplates}
                        disabled={isSubmitting}
                        labelId="presentation-template-label"
                        describedById="presentation-template-hint"
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
                            aria-describedby="presentation-language-hint"
                            className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 shadow-sm transition-colors focus-visible:border-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {PRESENTATION_LANGUAGES.map((code) => (
                                <option key={code} value={code}>{t(LANGUAGE_LABEL_KEYS[code])}</option>
                            ))}
                        </select>
                        <p id="presentation-language-hint" className="mt-1.5 text-xs text-slate-500">
                            {t('presentations.languageHint')}
                        </p>
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

                {validationError || submitError ? (
                    <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                        {validationError || submitError}
                    </p>
                ) : null}

                <div className="flex justify-end">
                    <Button
                        type="submit"
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
