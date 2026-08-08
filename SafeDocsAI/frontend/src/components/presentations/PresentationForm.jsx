import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowLeft, ArrowRight, RefreshCw, Sparkles, X } from 'lucide-react';

import PresentationTemplateGallery from './PresentationTemplateGallery';
import { Button } from '../ui/Button';
import Input from '../ui/Input';
import { useModalDialog } from '../../hooks/useModalDialog';
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
 * Шаги окна и их порядок.
 *
 * Порядок задан заказчиком и он же — единственный разумный: оформление
 * выбирается ПЕРВЫМ, параметры вторым. Шаблон виден целиком (превью занимает
 * шаг), а язык, число слайдов и описание уточняют уже выбранное оформление.
 *
 * Массивом, а не двумя булевыми флагами: индекс шага сразу даёт и «Шаг N из M»
 * в шапке, и направление кнопок «Назад»/«Далее», и его не приходится держать
 * в согласии со второй переменной.
 */
const STEPS = ['template', 'options'];

/**
 * Окно «Новая презентация»: выбор оформления, затем параметры заказа.
 *
 * Раньше это была форма, постоянно занимавшая пол-экрана над списком колод —
 * четыре поля, которые надо было пролистать, чтобы добраться до готовых
 * презентаций. Теперь на странице стоит одна кнопка, а всё, что нужно заказу,
 * живёт здесь и открывается по требованию.
 *
 * Диалог — тот же, что у остальных окон проекта (useModalDialog): фокус
 * переносится внутрь и возвращается на кнопку-инициатор, Escape закрывает, Tab
 * не уходит за пределы. Третьего способа делать модальные окна в проекте нет и
 * заводить его здесь незачем.
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
    isOpen,
    onClose,
    templates,
    templatesLoading,
    templatesError,
    onReloadTemplates,
    onSubmit,
    isSubmitting,
    submitError,
}) => {
    const { locale, t } = useLocale();

    const dialogRef = useRef(null);
    const closeButtonRef = useRef(null);

    const [stepIndex, setStepIndex] = useState(0);
    const [templateKey, setTemplateKey] = useState('');
    // Язык колоды по умолчанию — язык интерфейса: чаще всего заказывают на том,
    // на котором работают. Дальше значение живёт своей жизнью и переключением
    // локали не сбрасывается — иначе выбранный вручную язык молча менялся бы.
    const [language, setLanguage] = useState(() => localeToPresentationLanguage(locale));
    const [slideCount, setSlideCount] = useState(String(DEFAULT_SLIDE_COUNT));
    const [description, setDescription] = useState('');
    const [validationError, setValidationError] = useState('');

    // Открытие возвращает окно в начало: незакрытый прошлый заказ не должен
    // просвечивать в новом — ни шагом, ни набранным описанием. Сбрасывает
    // именно ОТКРЫТИЕ, а не закрытие: пока окно уезжает, показывать пустые
    // поля не нужно.
    useEffect(() => {
        if (!isOpen) return;

        setStepIndex(0);
        setTemplateKey('');
        setLanguage(localeToPresentationLanguage(locale));
        setSlideCount(String(DEFAULT_SLIDE_COUNT));
        setDescription('');
        setValidationError('');
        // locale намеренно не в зависимостях: язык колоды подставляется в
        // момент открытия, а переключение локали при открытом окне не должно
        // перебивать уже сделанный выбор.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    // Пока заказ в полёте, закрывать нечего: ответ всё равно придёт, а отказ
    // пользователь бы уже не увидел — окно с ним закрылось бы у него на глазах.
    const handleClose = useCallback(() => {
        if (isSubmitting) return;
        onClose?.();
    }, [isSubmitting, onClose]);

    useModalDialog(isOpen, handleClose, dialogRef, closeButtonRef);

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

    const selectedTemplateName = resolveTemplateName(
        templates.find((template) => template.key === selectedTemplateKey),
        locale,
    );

    const step = STEPS[stepIndex];

    const handleSubmit = (event) => {
        event.preventDefault();
        if (isSubmitting || !selectedTemplateKey) return;

        // На шаге оформления главная кнопка (и Enter) означают «дальше», а не
        // «сгенерировать»: отправить заказ, не показав параметры, — ровно то,
        // от чего заказчик и уходил. Шаги идут строго по порядку, поэтому
        // достаточно одной развилки, а не отдельного обработчика на кнопку.
        if (step !== 'options') {
            setStepIndex(STEPS.indexOf('options'));
            return;
        }

        // Отказ проверки возвращает на шаг параметров: поле, о котором говорит
        // сообщение, иначе осталось бы на другом шаге, и исправить его было бы
        // негде.
        const reject = (message) => {
            setValidationError(message);
            setStepIndex(STEPS.indexOf('options'));
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

    if (!isOpen) return null;

    const stepTitle = step === 'template'
        ? t('presentations.stepTemplateTitle')
        : t('presentations.stepOptionsTitle');

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4">
            <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" onClick={handleClose} aria-hidden="true" />

            {/* max-h + прокрутка внутри тела: на телефоне окно обязано помещаться
                целиком вместе с кнопками — уехавшая за нижний край «Сгенерировать»
                означала бы, что заказ сделать нечем. */}
            <div
                ref={dialogRef}
                role="dialog"
                aria-modal="true"
                aria-labelledby="presentation-dialog-title"
                className="relative flex max-h-[92vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
            >
                <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4 sm:px-6">
                    <div className="min-w-0">
                        <h2 id="presentation-dialog-title" className="text-base font-semibold text-slate-900 sm:text-lg">
                            {t('presentations.formTitle')}
                        </h2>
                        <p className="mt-1 text-xs font-semibold uppercase tracking-[0.12em] text-[#1f3a60]">
                            {t('presentations.stepOf', { step: stepIndex + 1, total: STEPS.length })} · {stepTitle}
                        </p>
                    </div>
                    <button
                        ref={closeButtonRef}
                        type="button"
                        onClick={handleClose}
                        disabled={isSubmitting}
                        aria-label={t('presentations.close')}
                        className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-60"
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>

                {/* noValidate — не отказ от проверки, а отказ от ЧУЖОЙ проверки:
                    браузер показывал своё «Value must be less than or equal to
                    15» по-английски поверх таджикского интерфейса и раньше
                    нашего переведённого сообщения. Диапазон проверяется ниже, в
                    handleSubmit, а min/max на поле остаются — они нужны
                    стрелкам ввода и вспомогательным технологиям. */}
                <form onSubmit={handleSubmit} noValidate className="flex min-h-0 flex-1 flex-col">
                    <div className="scrollbar-soft min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-5 sm:px-6">
                        {/* Почему заказ сейчас невозможен — НАД галереей: причина и
                            «Повторить» не должны прятаться на другом шаге, пока
                            «Далее» стоит выключенным без объяснения. Внутри галереи
                            этих же состояний нет — один экземпляр сообщения, а не
                            два расходящихся. */}
                        {step === 'template' && templatesError ? (
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

                        {step === 'template' && !templatesError && !templatesLoading && templates.length === 0 ? (
                            <p className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
                                {t('presentations.templatesEmpty')}
                            </p>
                        ) : null}

                        {step === 'template' ? (
                            <div>
                                <p id="presentation-template-label" className="mb-1 text-sm font-semibold text-slate-700">
                                    {t('presentations.templateLabel')}
                                </p>
                                <p className="mb-3 text-xs leading-5 text-slate-500">{t('presentations.templateHint')}</p>
                                <PresentationTemplateGallery
                                    templates={templates}
                                    selectedKey={selectedTemplateKey}
                                    onSelect={setTemplateKey}
                                    isLoading={templatesLoading}
                                    disabled={isSubmitting}
                                    labelId="presentation-template-label"
                                />
                            </div>
                        ) : null}

                        {step === 'options' ? (
                            <>
                                {/* Выбранное оформление остаётся на глазах и на втором
                                    шаге: параметры уточняют именно его, а возвращаться
                                    «посмотреть, что я выбрал» пришлось бы шагом назад. */}
                                {selectedTemplateName ? (
                                    <p className="inline-flex max-w-full items-center gap-2 rounded-full bg-[#1f3a60]/10 px-3 py-1.5 text-xs font-semibold text-[#1f3a60]">
                                        <Sparkles className="h-3.5 w-3.5 shrink-0" />
                                        <span className="min-w-0 break-words">
                                            {t('presentations.selectedTemplate', { name: selectedTemplateName })}
                                        </span>
                                    </p>
                                ) : null}

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

                                {/* «Несколько минут» названы ДО нажатия: заказ отвечает
                                    мгновенно, а делается долго, и узнавать об этом лучше
                                    не из ожидания. */}
                                <p className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-500">
                                    {t('presentations.formDescription')}
                                </p>
                            </>
                        ) : null}

                        {validationError || submitError ? (
                            <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                                {validationError || submitError}
                            </p>
                        ) : null}
                    </div>

                    {/* Главная кнопка ОДНА на оба шага и всегда type="submit" —
                        меняются только подпись и то, что делает handleSubmit.
                        Это не украшение, а защита от вполне конкретной ловушки:
                        React сохраняет DOM-узел кнопки между шагами (позиция в
                        дереве та же), и обработчик клика, переключающий шаг,
                        успевает поменять её type ПРЯМО ВО ВРЕМЯ обработки
                        нажатия — после чего браузер выполняет действие по
                        умолчанию уже для нового типа. «Далее» с type="button"
                        так отправляла форму одним кликом: заказ уходил на
                        сервер, минуя шаг параметров. Раз тип не меняется —
                        меняться нечему.

                        Побочная выгода: Enter работает на обоих шагах и делает
                        ровно то, что написано на кнопке. */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 px-5 py-4 sm:px-6">
                        <Button
                            type="button"
                            variant="ghost"
                            disabled={isSubmitting}
                            onClick={stepIndex === 0
                                ? handleClose
                                : () => setStepIndex(STEPS.indexOf('template'))}
                        >
                            {stepIndex === 0 ? t('presentations.cancel') : (
                                <>
                                    <ArrowLeft className="h-4 w-4" />
                                    {t('presentations.back')}
                                </>
                            )}
                        </Button>

                        <Button
                            type="submit"
                            isLoading={isSubmitting}
                            disabled={!selectedTemplateKey}
                            title={!selectedTemplateKey ? t('presentations.submitDisabledHint') : undefined}
                        >
                            {step === 'options' ? (
                                <>
                                    <Sparkles className="h-4 w-4" />
                                    {isSubmitting ? t('presentations.submitting') : t('presentations.submit')}
                                </>
                            ) : (
                                <>
                                    {t('presentations.next')}
                                    <ArrowRight className="h-4 w-4" />
                                </>
                            )}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default PresentationForm;
