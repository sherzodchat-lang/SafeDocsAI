import React from 'react';

import { cn } from '../../lib/utils';
import { useLocale } from '../../i18n';

const LOCALE_OPTIONS = [
    { value: 'tg', label: 'TG' },
    { value: 'ru', label: 'RU' },
];

const LocaleSwitcher = ({ className, buttonClassName }) => {
    const { locale, setLocale } = useLocale();

    return (
        <div className={cn('inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 p-1 text-xs font-semibold', className)}>
            {LOCALE_OPTIONS.map((option) => (
                <button
                    key={option.value}
                    type="button"
                    onClick={() => setLocale(option.value)}
                    data-active={locale === option.value}
                    className={cn(
                        'rounded-md px-3 py-1 transition',
                        locale === option.value
                            ? 'bg-white text-[#1f3a60] shadow-sm'
                            : 'text-slate-500 hover:text-slate-700',
                        buttonClassName,
                    )}
                    aria-pressed={locale === option.value}
                >
                    {option.label}
                </button>
            ))}
        </div>
    );
};

export default LocaleSwitcher;
