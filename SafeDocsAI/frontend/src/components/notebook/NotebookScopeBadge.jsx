import React from 'react';

import { useLocale } from '../../i18n';

/**
 * Бейдж текущей области данных и сброс к «всем источникам».
 *
 * Один компонент на чат и на список источников: оба экрана сужаются одним и тем
 * же активным блокнотом, поэтому и называть область они обязаны одинаково —
 * иначе пользователь опять получит фильтр, о котором нигде не сказано.
 *
 * Возвращает два элемента без обёртки: раскладку задаёт вызывающий экран, у
 * которого рядом с бейджем стоят собственные элементы шапки.
 */
const NotebookScopeBadge = ({ notebookId, notebookName, canReset = false, onReset, resetTitle }) => {
    const { t } = useLocale();
    const scopeValue = notebookId == null
        ? t('chat.allSources')
        : notebookName || `#${notebookId}`;

    return (
        <>
            <div className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-semibold text-[#1f3a60]">
                {t('chat.scopeLabel', { value: scopeValue })}
            </div>
            {/* Сброс — рядом с самой областью. Кнопки нет там, где область задана
                маршрутом или уже охватывает все источники: сбрасывать нечего. */}
            {canReset ? (
                <button
                    type="button"
                    onClick={onReset}
                    title={resetTitle}
                    className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
                >
                    {t('chat.scopeReset')}
                </button>
            ) : null}
        </>
    );
};

export default NotebookScopeBadge;
